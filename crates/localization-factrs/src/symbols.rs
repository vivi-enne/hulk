use factrs::{
    assign_symbols,
    variables::{ImuBias, SE23},
};

assign_symbols!(
    State: SE23;
    B: ImuBias;
);
