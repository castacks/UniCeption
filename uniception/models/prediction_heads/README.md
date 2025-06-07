# UniCeption Prediction Heads

## Currently Implemented Pathways

```
IntermediateFeatureReturner
├── DPTFeature
│   ├── DPTRegressionProcessor
│   │   ├── FlowAdaptor
│   │   ├── DepthAdaptor
|   │   ├── PointMapAdaptor
│   │   ├── ConfidenceAdaptor
│   │   ├── ValueWithConfidenceAdaptor
│   │   └── FlowWithConfidenceAdaptor
│   │   └── PointMapWithConfidenceAdaptor
│   └── DPTSegmentationProcessor
│       └── MaskAdaptor
└── LinearFeature
    └── ...(all adaptors)
```

The diagram outlines how implemented classes are designed to interact with each other.

## Developer Guidelines

Please follow the main UniCeption developer guidelines described in `README.md` when contributing to the prediction heads. Make sure to test your different implementations and add necessary unit tests.

## Happy Coding!
