# X3 macro diff

`results/x3/numbers_pre_x3.tex` -> `paper/numbers.tex`. Floors: 0.0664 for GSM8K macros, 0.0994 elsewhere.

| verdict | count |
|---|---|
| BEYOND FLOOR | 6 |
| not floor-testable | 22 |
| within floor | 90 |
| added | 7 |
| unchanged | 113 |

| macro | old | new | abs delta | verdict |
|---|---|---|---|---|
| `numCompAccVsNeff` | -0.702 | -0.506 | 0.1960 | BEYOND FLOOR |
| `numBanditSacValueNoise` | 0.138 | 0.300 | 0.1620 | BEYOND FLOOR |
| `numRealSepMathfiveFseven` | +0.372 | +0.528 | 0.1560 | BEYOND FLOOR |
| `numBanditSacAccuracy` | 0.177 | 0.043 | 0.1340 | BEYOND FLOOR |
| `numBanditSacValueCoherent` | 0.859 | 0.991 | 0.1320 | BEYOND FLOOR |
| `numCompAccVsAccMean` | +0.769 | +0.658 | 0.1110 | BEYOND FLOOR |
| `numCostInferenceRatio` | 5073 | 6598 | 1525.0000 | not floor-testable |
| `numHonestQMcItems` | 337 | 847 | 510.0000 | not floor-testable |
| `numCompletionTokensMean` | 493 | 634 | 141.0000 | not floor-testable |
| `numCostGeometricMedian` | 671.6 | 733.1 | 61.5000 | not floor-testable |
| `numCostSlowest` | 972 | 961 | 11.0000 | not floor-testable |
| `numHonestQMcPairs` | 9 | 19 | 10.0000 | not floor-testable |
| `numCostCoordMedian` | 75.6 | 83.8 | 8.2000 | not floor-testable |
| `numCostTrimmedMean` | 68.5 | 61.8 | 6.7000 | not floor-testable |
| `numCostMultiKrum` | 92.9 | 87.3 | 5.6000 | not floor-testable |
| `numAdversaryInvertedMean` | 76.3 | 80.6 | 4.3000 | not floor-testable |
| `numCostAipTrustOnly` | 77.8 | 74.4 | 3.4000 | not floor-testable |
| `numAucBelowChanceCells` | 21 | 18 | 3.0000 | not floor-testable |
| `numAucCellsPowered` | 22 | 19 | 3.0000 | not floor-testable |
| `numCostAipGated` | 79.6 | 77.1 | 2.5000 | not floor-testable |
| `numCostGateOverheadPct` | 2.2 | 3.6 | 1.4000 | not floor-testable |
| `numCostGateOverhead` | +1.7 | +2.7 | 1.0000 | not floor-testable |
| `numNeffUpperBoundMax` | 6.86 | 6.22 | 0.6400 | not floor-testable |
| `numCostKrum` | 20.9 | 21.2 | 0.3000 | not floor-testable |
| `numCompNeffBest` | 2.36 | 2.45 | 0.0900 | not floor-testable |
| `numPhiRatio` | 1.47 | 1.54 | 0.0700 | not floor-testable |
| `numNeffHomMax` | 1.91 | 1.87 | 0.0400 | not floor-testable |
| `numNeffHetMax` | 2.46 | 2.49 | 0.0300 | not floor-testable |
| `numSybilDisagreeingAip` | 0.662 | 0.748 | 0.0860 | within floor |
| `numSybilCoherentAip` | 0.672 | 0.756 | 0.0840 | within floor |
| `numCompRangeMathfive` | 0.248 | 0.170 | 0.0780 | within floor |
| `numNeffHomMin` | 1.07 | 1.14 | 0.0700 | within floor |
| `numCompBestGainMathfive` | -0.097 | -0.140 | 0.0430 | within floor |
| `numFalsifiedGsmkSelfreport` | -0.043 | -0.000 | 0.0430 | within floor |
| `numMaeMathfiveBinary` | 0.124 | 0.081 | 0.0430 | within floor |
| `numCompNeffWorst` | 1.42 | 1.38 | 0.0400 | within floor |
| `numSleeperDropMajority` | 0.032 | -0.004 | 0.0360 | within floor |
| `numQhatOpenTextMean` | 0.069 | 0.104 | 0.0350 | within floor |
| `numAucSelfReport` | 0.661 | 0.628 | 0.0330 | within floor |
| `numAucSelfReportFalsified` | 0.339 | 0.372 | 0.0330 | within floor |
| `numCompRangeMedqa` | 0.255 | 0.285 | 0.0300 | within floor |
| `numHoneFallMedqa` | -0.376 | -0.406 | 0.0300 | within floor |
| `numNeffHomMean` | 1.50 | 1.47 | 0.0300 | within floor |
| `numCompRangeMmlu` | 0.128 | 0.103 | 0.0250 | within floor |
| `numCompNeffVsAccMean` | -0.958 | -0.934 | 0.0240 | within floor |
| `numAucLogprob` | 0.724 | 0.704 | 0.0200 | within floor |
| `numMaeMathfiveMc` | 0.013 | 0.032 | 0.0190 | within floor |
| `numHoneFallMmlu` | -0.321 | -0.338 | 0.0170 | within floor |
| `numMaeMedqaMc` | 0.047 | 0.030 | 0.0170 | within floor |
| `numAttackErrRushing` | 0.958 | 0.974 | 0.0160 | within floor |
| `numBurstRecoveredTwenty` | +0.195 | +0.210 | 0.0150 | within floor |
| `numBurstRecoveredTen` | +0.191 | +0.205 | 0.0140 | within floor |
| `numSleeperDropAip` | 0.150 | 0.136 | 0.0140 | within floor |
| `numAucGap` | +0.063 | +0.076 | 0.0130 | within floor |
| `numHonestQMc` | 0.560 | 0.547 | 0.0130 | within floor |
| `numWithinModelPhi` | 0.652 | 0.665 | 0.0130 | within floor |
| `numBurstRecoveredFive` | +0.188 | +0.200 | 0.0120 | within floor |
| `numCrossModelPhi` | 0.444 | 0.432 | 0.0120 | within floor |
| `numHonestQMathfive` | 0.015 | 0.027 | 0.0120 | within floor |
| `numMaeMmluMc` | 0.073 | 0.061 | 0.0120 | within floor |
| `numRealSepMedqaFseven` | +0.008 | +0.020 | 0.0120 | within floor |
| `numRealSepMmluFseven` | +0.188 | +0.200 | 0.0120 | within floor |
| `numAttackQRushing` | 0.146 | 0.156 | 0.0100 | within floor |
| `numBurstStatic` | 0.776 | 0.766 | 0.0100 | within floor |
| `numSelfReportMissingMax` | 0.340 | 0.330 | 0.0100 | within floor |
| `numHoneFallArc` | -0.316 | -0.307 | 0.0090 | within floor |
| `numHoneFallBoolq` | -0.615 | -0.606 | 0.0090 | within floor |
| `numMaeBoolqBinary` | 0.092 | 0.101 | 0.0090 | within floor |
| `numSleeperDropReputation` | 0.040 | 0.031 | 0.0090 | within floor |
| `numCompRangeGsmk` | 0.029 | 0.021 | 0.0080 | within floor |
| `numMaeMmluBinary` | 0.087 | 0.079 | 0.0080 | within floor |
| `numSleeperDropAipWindowed` | 0.129 | 0.121 | 0.0080 | within floor |
| `numMaeArcBinary` | 0.020 | 0.027 | 0.0070 | within floor |
| `numMaeMedqaBinary` | 0.048 | 0.041 | 0.0070 | within floor |
| `numParityAipTieMean` | 0.205 | 0.212 | 0.0070 | within floor |
| `numGsmkNegationGain` | +0.050 | +0.044 | 0.0060 | within floor |
| `numSybilDisagreeingBaselines` | 0.277 | 0.283 | 0.0060 | within floor |
| `numBanditAipValueNoise` | 0.138 | 0.133 | 0.0050 | within floor |
| `numBurstWindowTwenty` | 0.970 | 0.975 | 0.0050 | within floor |
| `numCompBestGainArc` | +0.015 | +0.010 | 0.0050 | within floor |
| `numCompBestGainMmlu` | +0.002 | +0.007 | 0.0050 | within floor |
| `numGsmkHallucinationGain` | +0.027 | +0.032 | 0.0050 | within floor |
| `numHoneFallMathfive` | -0.081 | -0.086 | 0.0050 | within floor |
| `numMmluNegQhat` | 0.320 | 0.325 | 0.0050 | within floor |
| `numRealDiscardMaxFseven` | 0.595 | 0.600 | 0.0050 | within floor |
| `numRealSepArcFseven` | +0.268 | +0.273 | 0.0050 | within floor |
| `numRealSepBoolqFseven` | -0.290 | -0.295 | 0.0050 | within floor |
| `numSleeperDropSpread` | 0.110 | 0.105 | 0.0050 | within floor |
| `numAttackErrAlwaysWrong` | 0.678 | 0.682 | 0.0040 | within floor |
| `numAttackQHallucination` | 0.062 | 0.066 | 0.0040 | within floor |
| `numBurstWindowTen` | 0.966 | 0.970 | 0.0040 | within floor |
| `numFalsifiedMmluSelfreport` | -0.159 | -0.163 | 0.0040 | within floor |
| `numGsmkHallucinationQ` | 0.062 | 0.066 | 0.0040 | within floor |
| `numMaeArcMc` | 0.013 | 0.017 | 0.0040 | within floor |
| `numQhatOpenNumericMean` | 0.246 | 0.250 | 0.0040 | within floor |
| `numBanditAipAccuracy` | 0.863 | 0.866 | 0.0030 | within floor |
| `numBanditAipValueCoherent` | 0.137 | 0.134 | 0.0030 | within floor |
| `numBurstWindowFive` | 0.963 | 0.966 | 0.0030 | within floor |
| `numCompBestGainMedqa` | +0.015 | +0.018 | 0.0030 | within floor |
| `numMaeBoolqMc` | 0.076 | 0.079 | 0.0030 | within floor |
| `numMaeGsmkBinary` | 0.029 | 0.026 | 0.0030 | within floor |
| `numAttackErrHallucination` | 0.978 | 0.980 | 0.0020 | within floor |
| `numAttackQAlwaysWrong` | 0.100 | 0.102 | 0.0020 | within floor |
| `numFloorGsmk` | 0.068 | 0.066 | 0.0020 | within floor |
| `numHoneFallGsmk` | -0.012 | -0.014 | 0.0020 | within floor |
| `numQhatBinaryMean` | 0.557 | 0.559 | 0.0020 | within floor |
| `numQhatMcMean` | 0.283 | 0.285 | 0.0020 | within floor |
| `numSybilCoherentBaselines` | 0.265 | 0.267 | 0.0020 | within floor |
| `numCompBestGainGsmk` | +0.012 | +0.013 | 0.0010 | within floor |
| `numFalsifiedGsmkAip` | +0.000 | -0.001 | 0.0010 | within floor |
| `numFloorArc` | 0.100 | 0.099 | 0.0010 | within floor |
| `numFloorBoolq` | 0.100 | 0.099 | 0.0010 | within floor |
| `numFloorMathfive` | 0.100 | 0.099 | 0.0010 | within floor |
| `numFloorMedqa` | 0.100 | 0.099 | 0.0010 | within floor |
| `numFloorMmlu` | 0.100 | 0.099 | 0.0010 | within floor |
| `numFloorWorst` | 0.100 | 0.099 | 0.0010 | within floor |
| `numMaeGsmkMc` | 0.019 | 0.018 | 0.0010 | within floor |
| `numParityAipGated` | 0.0279 | 0.0283 | 0.0004 | within floor |
| `numBudgetTokens` | -- | 3072 |  | added |
| `numCacheRows` | -- | 13500 |  | added |
| `numManufacturedAfter` | -- | 356 |  | added |
| `numManufacturedBefore` | -- | 944 |  | added |
| `numManufacturedMathfive` | -- | 0.351 |  | added |
| `numManufacturedRateBefore` | -- | 0.070 |  | added |
| `numTruncPinnedBefore` | -- | 1036 |  | added |
| `numAgents` | 10 | 10 |  | unchanged |
| `numAttackClasses` | 7 | 7 |  | unchanged |
| `numAttackErrBurst` | 0.996 | 0.996 |  | unchanged |
| `numAttackErrFalsified` | 0.996 | 0.996 |  | unchanged |
| `numAttackErrNegation` | 0.996 | 0.996 |  | unchanged |
| `numAttackQBurst` | 0.674 | 0.674 |  | unchanged |
| `numAttackQFalsified` | 0.674 | 0.674 |  | unchanged |
| `numAttackQNegation` | 0.674 | 0.674 |  | unchanged |
| `numAucCellsTotal` | 42 | 42 |  | unchanged |
| `numBenchmarks` | 6 | 6 |  | unchanged |
| `numBootstrapResamples` | 500 | 500 |  | unchanged |
| `numCompBestGainBoolq` | +0.015 | +0.015 |  | unchanged |
| `numCompRangeArc` | 0.043 | 0.043 |  | unchanged |
| `numCompRangeBoolq` | 0.045 | 0.045 |  | unchanged |
| `numCompSubsetsFive` | 21 | 21 |  | unchanged |
| `numCompSubsetsThree` | 35 | 35 |  | unchanged |
| `numCostMajority` | 3.2 | 3.2 |  | unchanged |
| `numDiscardCollapseFseven` | 0.000 | 0.000 |  | unchanged |
| `numDiscardOracleBoolqFseven` | 0.815 | 0.815 |  | unchanged |
| `numDiscardOracleMedqaFseven` | 0.745 | 0.745 |  | unchanged |
| `numEvasionAccMathfive` | 0.366 | 0.366 |  | unchanged |
| `numEvasionAccMedqa` | 0.082 | 0.082 |  | unchanged |
| `numEvasionAccMmlu` | 0.374 | 0.374 |  | unchanged |
| `numEvasionAnchorF` | 0.5 | 0.5 |  | unchanged |
| `numEvasionBandHi` | 0.40 | 0.40 |  | unchanged |
| `numEvasionBandLo` | 0.15 | 0.15 |  | unchanged |
| `numEvasionChance` | 0.333 | 0.333 |  | unchanged |
| `numEvasionCoherentMathfive` | 0.000 | 0.000 |  | unchanged |
| `numEvasionCoherentMedqa` | 0.878 | 0.878 |  | unchanged |
| `numEvasionCoherentMmlu` | 0.903 | 0.903 |  | unchanged |
| `numEvasionGainMathfive` | -0.366 | -0.366 |  | unchanged |
| `numEvasionGainMedqa` | +0.796 | +0.796 |  | unchanged |
| `numEvasionGainMmlu` | +0.529 | +0.529 |  | unchanged |
| `numEvasionQDistinct` | 3 | 3 |  | unchanged |
| `numEvasionQFloorMc` | 0.253 | 0.253 |  | unchanged |
| `numEvasionQMathfive` | 0.160 | 0.160 |  | unchanged |
| `numEvasionQMedqa` | 0.280 | 0.280 |  | unchanged |
| `numEvasionQMmlu` | 0.280 | 0.280 |  | unchanged |
| `numFalsifiedMmluAip` | +0.000 | +0.000 |  | unchanged |
| `numGsmkAipFseven` | 0.976 | 0.976 |  | unchanged |
| `numGsmkAipFzero` | 0.976 | 0.976 |  | unchanged |
| `numGsmkAipMax` | 0.984 | 0.984 |  | unchanged |
| `numGsmkAipMin` | 0.976 | 0.976 |  | unchanged |
| `numGsmkAipSpread` | 0.008 | 0.008 |  | unchanged |
| `numGsmkHonestFloor` | 0.597 | 0.597 |  | unchanged |
| `numGsmkNegQhat` | 0.674 | 0.674 |  | unchanged |
| `numGsmkNegationQ` | 0.674 | 0.674 |  | unchanged |
| `numHeadlineBenchmarks` | 3 | 3 |  | unchanged |
| `numHonestInvertedMax` | 0.0 | 0.0 |  | unchanged |
| `numInversionCostFzero` | 0.001 | 0.001 |  | unchanged |
| `numInversionGainCompetenceDiff` | +0.175 | +0.175 |  | unchanged |
| `numInversionGainFseven` | 0.645 | 0.645 |  | unchanged |
| `numInversionGainNFifty` | 0.655 | 0.655 |  | unchanged |
| `numInversionGainNRange` | 0.018 | 0.018 |  | unchanged |
| `numInversionGainNTen` | 0.643 | 0.643 |  | unchanged |
| `numInversionGainNTwenty` | 0.637 | 0.637 |  | unchanged |
| `numInversionGainWeak` | 0.470 | 0.470 |  | unchanged |
| `numInversionGainWeakAllBenches` | 0.547 | 0.547 |  | unchanged |
| `numMathfiveAipFseven` | 0.000 | 0.000 |  | unchanged |
| `numMathfiveAipFzero` | 0.477 | 0.477 |  | unchanged |
| `numMathfiveAipMax` | 0.477 | 0.477 |  | unchanged |
| `numMathfiveAipMin` | 0.000 | 0.000 |  | unchanged |
| `numMathfiveAipSpread` | 0.477 | 0.477 |  | unchanged |
| `numMitigBothBandDelta` | +0.174 | +0.174 |  | unchanged |
| `numMitigBothBandMean` | 0.638 | 0.638 |  | unchanged |
| `numMitigBothBandWorst` | 0.278 | 0.278 |  | unchanged |
| `numMitigBothDelta` | +0.109 | +0.109 |  | unchanged |
| `numMitigBothMean` | 0.566 | 0.566 |  | unchanged |
| `numMitigHardBandMean` | 0.463 | 0.463 |  | unchanged |
| `numMitigHardBandWorst` | 0.082 | 0.082 |  | unchanged |
| `numMitigHardMean` | 0.457 | 0.457 |  | unchanged |
| `numMitigRandBandDelta` | +0.010 | +0.010 |  | unchanged |
| `numMitigRandBandMean` | 0.474 | 0.474 |  | unchanged |
| `numMitigRandBandWorst` | 0.082 | 0.082 |  | unchanged |
| `numMitigRandDelta` | -0.013 | -0.013 |  | unchanged |
| `numMitigRandMean` | 0.444 | 0.444 |  | unchanged |
| `numMitigSoftBandDelta` | +0.160 | +0.160 |  | unchanged |
| `numMitigSoftBandMean` | 0.623 | 0.623 |  | unchanged |
| `numMitigSoftBandWorst` | 0.200 | 0.200 |  | unchanged |
| `numMitigSoftDelta` | +0.101 | +0.101 |  | unchanged |
| `numMitigSoftMean` | 0.557 | 0.557 |  | unchanged |
| `numMmluAipFseven` | 0.902 | 0.902 |  | unchanged |
| `numMmluAipFzero` | 0.888 | 0.888 |  | unchanged |
| `numMmluAipMax` | 0.920 | 0.920 |  | unchanged |
| `numMmluAipMin` | 0.877 | 0.877 |  | unchanged |
| `numMmluAipSpread` | 0.043 | 0.043 |  | unchanged |
| `numMmluChanceQ` | 0.333 | 0.333 |  | unchanged |
| `numMmluHonestFloor` | 0.535 | 0.535 |  | unchanged |
| `numModelAccMax` | 0.930 | 0.930 |  | unchanged |
| `numModelAccMin` | 0.340 | 0.340 |  | unchanged |
| `numModels` | 7 | 7 |  | unchanged |
| `numModelsAll` | 9 | 9 |  | unchanged |
| `numModelsWeakArm` | 2 | 2 |  | unchanged |
| `numParamsMaxB` | 32 | 32 |  | unchanged |
| `numParamsMinB` | 3 | 3 |  | unchanged |
| `numParityAipOffTieMax` | 0.135 | 0.135 |  | unchanged |
| `numParityAipTieMax` | 0.840 | 0.840 |  | unchanged |
| `numParityUnweightedMax` | 0.0000 | 0.0000 |  | unchanged |
| `numPobsFullGsmkFsix` | 0.980 | 0.980 |  | unchanged |
| `numPobsHalfGsmkFsix` | 0.917 | 0.917 |  | unchanged |
| `numPobsQuarter` | 0.25 | 0.25 |  | unchanged |
| `numPobsTenth` | 0.10 | 0.10 |  | unchanged |
| `numRealDiscardMinFseven` | 0.002 | 0.002 |  | unchanged |
| `numRealSepGsmkFseven` | +0.857 | +0.857 |  | unchanged |
| `numRegretGsmkAip` | 0.004 | 0.004 |  | unchanged |
| `numRegretGsmkNext` | 0.963 | 0.963 |  | unchanged |
| `numRegretMathfiveAip` | 0.523 | 0.523 |  | unchanged |
| `numRegretMathfiveNext` | 0.438 | 0.438 |  | unchanged |
| `numRegretMmluAip` | 0.038 | 0.038 |  | unchanged |
| `numRegretMmluNext` | 0.902 | 0.902 |  | unchanged |
| `numSybilSizeMax` | 8 | 8 |  | unchanged |
| `numTasksMax` | 500 | 500 |  | unchanged |
| `numTasksMin` | 200 | 200 |  | unchanged |
