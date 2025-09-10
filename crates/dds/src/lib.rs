use std::{future::Future, time::Duration};

use booster_low_level_interface::{
    BoosterLowLevelInterface, ButtonEventMsg, FallDownState, LowCommand, LowState,
    RemoteControllerState, TransformStamped,
};
use color_eyre::Result;
use futures::{executor::block_on, stream::StreamExt};
use rustdds::{
    dds::ReadError,
    no_key::{DataReaderStream, DataSample, DataWriter},
    DomainParticipant, Publisher, QosPolicies, Subscriber, Topic,
};
use tokio::sync::broadcast::{self, Receiver, Sender};

const BUFFER_SIZE: usize = 100;
const FIND_TOPIC_TIMEOUT: Duration = Duration::from_secs(1);

struct TopicInfos {
    low_state: TopicInfo,
    joint_ctrl: TopicInfo,
    fall_down: TopicInfo,
    button_event: TopicInfo,
    remote_controller_state: TopicInfo,
    transform: TopicInfo,
}

impl TopicInfos {
    fn new() -> Self {
        Self {
            low_state: TopicInfo::new(
                "rt/low_state",
                "Obtain the robot's IMU and joint feedback in real time.",
            ),
            joint_ctrl: TopicInfo::new(
                "rt/joint_ctrl",
                "Publish the joint commands of the robot to control the motors.",
            ),
            fall_down: TopicInfo::new("rt/fall_down", "Real-time detection of robot falls"),
            button_event: TopicInfo::new(
                "rt/button_event",
                "Real-time retrieval of backboard button inputs",
            ),
            remote_controller_state: TopicInfo::new(
                "rt/remote_controller_state",
                "Real-time retrieval of remote controller button inputs",
            ),
            transform: TopicInfo::new(
                "rt/tf",
                "Real-time acquisition of coordinate transformations between robot joints",
            ),
        }
    }
}

struct TopicInfo {
    name: &'static str,
    description: &'static str,
}

impl TopicInfo {
    const fn new(name: &'static str, description: &'static str) -> Self {
        TopicInfo { name, description }
    }
}

pub struct DdsParticipant {
    participant: DomainParticipant,
    subscriber: Subscriber,
    publisher: Publisher,

    low_state_sender: Sender<LowState>,
    low_cmd_sender: Sender<LowCommand>,
    low_cmd_receiver: Receiver<LowCommand>,
    fall_down_state_sender: Sender<FallDownState>,
    button_event_msg_sender: Sender<ButtonEventMsg>,
    remote_controller_state_sender: Sender<RemoteControllerState>,
    transform_stamped_sender: Sender<TransformStamped>,
}

impl DdsParticipant {
    fn try_new(domain_id: u16) -> Result<Self> {
        let participant = DomainParticipant::new(domain_id)?;
        let subscriber = participant.create_subscriber(&QosPolicies::qos_none())?;
        let publisher = participant.create_publisher(&QosPolicies::qos_none())?;

        let (low_state_sender, _) = broadcast::channel::<LowState>(BUFFER_SIZE);
        let (low_cmd_sender, low_cmd_receiver) = broadcast::channel::<LowCommand>(BUFFER_SIZE);
        let (fall_down_state_sender, _) = broadcast::channel::<FallDownState>(BUFFER_SIZE);
        let (button_event_msg_sender, _) = broadcast::channel::<ButtonEventMsg>(BUFFER_SIZE);
        let (remote_controller_state_sender, _) =
            broadcast::channel::<RemoteControllerState>(BUFFER_SIZE);
        let (transform_stamped_sender, _) = broadcast::channel::<TransformStamped>(BUFFER_SIZE);

        Ok(Self {
            participant,
            subscriber,
            publisher,
            low_state_sender,
            low_cmd_sender,
            low_cmd_receiver,
            fall_down_state_sender,
            button_event_msg_sender,
            remote_controller_state_sender,
            transform_stamped_sender,
        })
    }

    fn start(mut self) -> Result<()> {
        dbg!("Started server");
        let topic_infos = TopicInfos::new();
        let mut low_state_data_stream: DataReaderStream<LowState> = self
            .subscriber
            .create_datareader_no_key_cdr(
                &find_or_create_topic(&self.participant, topic_infos.low_state),
                None,
            )?
            .async_sample_stream();
        let mut joint_ctrl_data_writer: DataWriter<LowCommand> =
            self.publisher.create_datawriter_no_key_cdr(
                &find_or_create_topic(&self.participant, topic_infos.joint_ctrl),
                None,
            )?;
        let mut fall_down_state_data_stream: DataReaderStream<FallDownState> = self
            .subscriber
            .create_datareader_no_key_cdr(
                &find_or_create_topic(&self.participant, topic_infos.fall_down),
                None,
            )?
            .async_sample_stream();
        let mut button_event_msg_data_stream: DataReaderStream<ButtonEventMsg> = self
            .subscriber
            .create_datareader_no_key_cdr(
                &find_or_create_topic(&self.participant, topic_infos.button_event),
                None,
            )?
            .async_sample_stream();
        let mut remote_controller_state_data_stream: DataReaderStream<RemoteControllerState> = self
            .subscriber
            .create_datareader_no_key_cdr(
                &find_or_create_topic(&self.participant, topic_infos.remote_controller_state),
                None,
            )?
            .async_sample_stream();
        let mut transform_stamped_data_stream: DataReaderStream<TransformStamped> = self
            .subscriber
            .create_datareader_no_key_cdr(
                &find_or_create_topic(&self.participant, topic_infos.transform),
                None,
            )?
            .async_sample_stream();

        let read_loop = async move {
            loop {
                tokio::select! {
                    received_sample = low_state_data_stream.select_next_some() => {
                        handle_received_dds_sample(received_sample);
                    },
                    received_sample = fall_down_state_data_stream.select_next_some() => {
                        handle_received_dds_sample(received_sample);
                    },
                    received_sample = button_event_msg_data_stream.select_next_some() => {
                        handle_received_dds_sample(received_sample);
                    },
                    received_sample = remote_controller_state_data_stream.select_next_some() => {
                        handle_received_dds_sample(received_sample);
                    },
                    received_sample = transform_stamped_data_stream.select_next_some() => {
                        handle_received_dds_sample(received_sample);
                    },
                }
            }
        };

        let write_loop = async move {
            loop {
                tokio::select! {
                    received = self.low_cmd_receiver.recv() => {
                        match received {
                            Ok(sample) => {
                                dbg!(sample);
                            },
                            Err(err) => {
                                dbg!(err);
                            },
                        }
                    }
                }
            }
        };

        tokio::spawn(async { futures::join!(read_loop, write_loop) });

        // let runtime = Runtime::new()?;
        // let low_state_sender = self.low_state_sender.clone();
        // runtime.spawn(async move {
        //     loop {
        //         for data in
        //             .conditional_iterator(ReadCondition::any())
        //             .unwrap()
        //         {
        //             low_state_sender.send(data.clone()).unwrap();
        //         }
        //     }
        // });
        // let fall_down_state_sender: Sender<FallDownState> = self.fall_down_state_sender.clone();
        // runtime.spawn(async move {
        //     loop {
        //         for data in fall_down_state_data_reader
        //             .conditional_iterator(ReadCondition::any())
        //             .unwrap()
        //         {
        //             fall_down_state_sender.send(data.clone()).unwrap();
        //         }
        //     }
        // });
        // let button_event_msg_sender = self.button_event_msg_sender.clone();
        // runtime.spawn(async move {
        //     loop {
        //         for data in button_event_msg_data_reader
        //             .conditional_iterator(ReadCondition::any())
        //             .unwrap()
        //         {
        //             button_event_msg_sender.send(data.clone()).unwrap();
        //         }
        //     }
        // });
        // let remote_controller_state_sender = self.remote_controller_state_sender.clone();
        // runtime.spawn(async move {
        //     loop {
        //         for data in remote_controller_state_data_reader
        //             .conditional_iterator(ReadCondition::any())
        //             .unwrap()
        //         {
        //             remote_controller_state_sender.send(data.clone()).unwrap();
        //         }
        //     }
        // });
        // let transform_stamped_sender = self.transform_stamped_sender.clone();
        // runtime.spawn(async move {
        //     loop {
        //         for data in transform_stamped_data_reader
        //             .conditional_iterator(ReadCondition::any())
        //             .unwrap()
        //         {
        //             transform_stamped_sender.send(data.clone()).unwrap();
        //         }
        //     }
        // });
        // let mut low_cmd_receiver = self.low_cmd_receiver;
        // runtime.spawn(async move {
        //     loop {
        //         if let Ok(cmd) = low_cmd_receiver.try_recv() {
        //             joint_ctrl_data_writer.write(cmd, None).unwrap();
        //         }
        //         time::sleep(Duration::from_millis(1)).await;
        //     }
        // });
        Ok(())
    }
}

fn handle_received_dds_sample<D: std::fmt::Debug>(sample_result: Result<DataSample<D>, ReadError>) {
    match sample_result {
        Ok(sample) => {
            dbg!(sample.into_value());
        }
        Err(err) => (),
    }
}

impl BoosterLowLevelInterface for DdsParticipant {
    fn subscribe_low_state(&self) -> Receiver<booster_low_level_interface::LowState> {
        self.low_state_sender.subscribe()
    }

    fn publish_joint_ctrl(&self) -> Sender<booster_low_level_interface::LowCommand> {
        let new_sender = self.low_cmd_sender.clone();
        assert!(new_sender.same_channel(&self.low_cmd_sender));
        new_sender
    }

    fn subscribe_fall_down(&self) -> Receiver<booster_low_level_interface::FallDownState> {
        self.fall_down_state_sender.subscribe()
    }

    fn subscribe_button_event(&self) -> Receiver<booster_low_level_interface::ButtonEventMsg> {
        self.button_event_msg_sender.subscribe()
    }

    fn subscribe_remote_controller_state(
        &self,
    ) -> Receiver<booster_low_level_interface::RemoteControllerState> {
        self.remote_controller_state_sender.subscribe()
    }

    fn subscribe_frame_transform(&self) -> Receiver<booster_low_level_interface::TransformStamped> {
        self.transform_stamped_sender.subscribe()
    }
}

fn find_or_create_topic(participant: &DomainParticipant, topic_info: TopicInfo) -> Topic {
    if let Some(topic) = participant
        .find_topic(topic_info.name, FIND_TOPIC_TIMEOUT)
        .ok()
        .flatten()
    {
        topic
    } else {
        dbg!("Creating topic..");
        participant
            .create_topic(
                "EEEEEELSE".to_string(),
                topic_info.description.to_string(),
                &QosPolicies::qos_none(),
                rustdds::TopicKind::NoKey,
            )
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;

    use booster_low_level_interface::MotorCommand;
    use rustdds::ReadCondition;

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn dds_publisher_subscriber() {
        let dummy_participant = DomainParticipant::new(0).expect("failed to create participant");
        let joint_ctrl_topic = dummy_participant
            .create_topic(
                "rt/joint_ctrl".to_string(),
                "".to_string(),
                &QosPolicies::qos_none(),
                rustdds::TopicKind::NoKey,
            )
            .expect("failed to create topic");

        let fall_down_state_topic = dummy_participant
            .create_topic(
                "rt/fallen_down_state".to_string(),
                "".to_string(),
                &QosPolicies::qos_none(),
                rustdds::TopicKind::NoKey,
            )
            .expect("failed to create topic");

        let dummy_subscriber = dummy_participant
            .create_subscriber(&QosPolicies::qos_none())
            .expect("failed to create subscriber");
        let dummy_publisher = dummy_participant
            .create_publisher(&QosPolicies::qos_none())
            .expect("failed tasynco create publisher");

        let mut dummy_data_reader = dummy_subscriber
            .create_datareader_no_key_cdr::<LowCommand>(&joint_ctrl_topic, None)
            .expect("failed to create data reader");
        let dummy_data_writer = dummy_publisher
            .create_datawriter_no_key_cdr::<FallDownState>(&fall_down_state_topic, None)
            .expect("failed to create data writer");

        let mut dds_participant = DdsParticipant::try_new(0).unwrap();
        let joint_ctrl_sender = dds_participant.publish_joint_ctrl();

        let mut fall_down_state_receiver = dds_participant.subscribe_fall_down();
        dds_participant.start();

        joint_ctrl_sender
            .send(LowCommand {
                command_type: booster_low_level_interface::CommandType::Parallel,
                motor_command: vec![MotorCommand {
                    position: 4.0,
                    velocity: 5.0,
                    torque: 1.0,
                    kp: 2.0,
                    kd: 42.0,
                    weight: 30.0,
                }],
            })
            .unwrap();

        dbg!(dummy_participant.discovered_topics());
        dbg!(dummy_data_reader
            .read_next_sample()
            .expect("failed to read data")
            .unwrap()
            .into_value());
        // assert!(
        //     dummy_data_reader
        //         .read_next_sample()
        //         .expect("failed to read data")
        //         .unwrap()
        //         .into_value()
        //         > 0
        // );

        dummy_data_writer
            .write(
                FallDownState {
                    fall_down_state: booster_low_level_interface::FallDownStateType::IsFalling,
                    is_recovery_available: false,
                },
                None,
            )
            .expect("failed to write data");
        assert!(fall_down_state_receiver.try_recv().is_ok());
    }
}
