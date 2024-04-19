
# Feature Engineering

Potential Predictive Features

| Name                   | Description                                                                   | Notes
|------------------------|-------------------------------------------------------------------------------|--------------
| distance from home     | How far transactor was from their home when they made the transaction         |
| distance to next closest transaction | How far transactor was from their next nearest historic transaction  |
| ratio to median amount | [transaction amount] / [median transaction amount]                            | Could use mean, max etc.
|                        |                                                                               | This will be noisy for first few transactions
| repeat source/destination count | Number of previous historic transactions at this source/destination  | 
| Time of day            | | 
| Day of week            | |
| Day in month           | |
| transaction location   | |
| payment method         | |
