export type Prediction = {
    // ID of the quest
    value: number;
    // Name of the quest
    quest: string;
    // Confidence of the model, from 0 to 1
    // Note that the worker filters result by confidence already. This is just FYI
    confidence: number;
}