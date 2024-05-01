import type { NamedTensorMap, Tensor } from "@tensorflow/tfjs-core";
import type { Prediction } from "./type";

declare const tf: typeof import ("@tensorflow/tfjs-core");
declare const tflite: typeof import ("@tensorflow/tfjs-tflite");

const outputElem = document.getElementById("output") as HTMLElement;
function print(text: string) {
    outputElem.innerText = text;
    console.log(`[worker] ${text}`);
}


let pingHost: number | undefined = undefined;

const QUESTS = [
    "<None>",
    "Follow the Sheikah Slate",
    "The Isolated Plateau",
    "Seek Out Impa",
    "The Hero's Sword",
    "Free the Divine Beasts",
    "Locked Mementos",
    "Divine Beast Vah Medoh",
    "Divine Beast Vah Rudania",
    "Reach Zora's Domain",
    "Divine Beast Vah Ruta",
    "Forbidden City Entry",
    "Divine Beast Vah Naboris",
    "Captured Memories",
    "Find the Fairy Fountain",
    "Destroy Ganon",
    "Robbie's Research",
    "From the Ground Up",
    "A Parent's Love",
    "Hobbies of the Rich",
    "A Shady Customer",
    "Little Sister's Big Request",
    "Hylian Homeowner",
    "The Statue's Bargain",
    "A Gift for My Beloved",
    "The Weapon Connoisseur",
    "The Sheep Rustlers",
    "Sunshroom Sensing",
    "Slated for Upgrades",
    "Sunken Treasure",
    "What's for Dinner?",
    "Take Back the Sea",
    "Koko's Kitchen",
    "Cooking with Koko",
    "Koko Cuisine",
    "Koko's Specialty",
    "Playtime with Cottla",
    "By Firefly's Light",
    "Flown the Coop",
    "The Priceless Maracas",
    "Arrows of Burning Heat",
    "Stalhorse: Pictured!",
    "Curry for What Ails You",
    "Find Kheel",
    "Face the Frost Talus",
    "The Apple of My Eye",
    "The Spark of Romance",
    "The Jewel Trade",
    "Death Mountain's Secret",
    "The Road to Respect",
    "Fireproof Lizard Roundup",
    "Balloon Flight",
    "The Thunder Helm",
    "The Search for Barta",
    "Medicinal Molduga",
    "The Eighth Heroine",
    "The Mystery Polluter",
    "The Secret Club's Secret",
    "Tools of the Trade",
    "The Forgotten Sword",
    "Missing in Action",
    "Rushroom Rush!",
    "Good-Sized Horse",
    "An Ice Guy",
    "A Freezing Rod",
    "The Korok Trials",
    "Riddles of Hyrule",
    "Legendary Rabbit Trial",
    "Special Delivery",
    "Lynel Safari",
    "The Giant of Ralis Pond",
    "Frog Catching",
    "Zora Stone Monuments",
    "Diving is Beauty!",
    "Luminous Stone Gathering",
    "A Wife Washed Away",
    "A Gift from the Monks",
    "The Hero's Cache",
    "Misko, the Great Bandit",
    "Wild Horses",
    "A Gift of Nightshade",
    "Hunt for the Giant Horse",
    "The Horseback Hoodlums",
    "Thunder Magnet",
    "A Gift for the Great Fairy",
    "Leviathan Bones",
    "The Royal Guard's Gear",
    "A Royal Recipe",
    "Riverbed Reward",
    "My Hero",
    "A Rare Find",
    "The Royal White Stallion",
    "[Xenoblade Chronicles 2]",
    "The Skull's Eye",
    "Into the Vortex",
    "Trial of the Labyrinth",
    "The Spring of Power",
    "The Gut Check Challenge",
    "A Brother's Roast",
    "A Landscape of a Stable",
    "The Perfect Drink",
    "Test of Will",
    "Sign of the Shadow",
    "The Silent Swordswomen",
    "The Desert Labyrinth",
    "The Seven Heroines",
    "The Eye of the Sandstorm",
    "Secret of the Snowy Peaks",
    "The Undefeated Champ",
    "Watch Out for the Flowers",
    "The Three Giant Brothers",
    "Secret of the Cedars",
    "The Cursed Statue",
    "A Fragmented Monument",
    "The Stolen Heirloom",
    "Guardian Slideshow",
    "A Song of Storms",
    "The Serpent's Jaws",
    "Stranded on Eventide",
    "The Bird in the Mountains",
    "Recital at Warbler's Nest",
    "The Ancient Rito Song",
    "Trial on the Cliff",
    "The Spring of Wisdom",
    "The Ceremonial Song",
    "The Crowned Beast",
    "Master of the Wind",
    "The Lost Pilgrimage",
    "The Two Rings",
    "Shrouded Shrine",
    "Under a Red Moon",
    "Cliffside Etchings",
    "Trial of Second Sight",
    "The Test of Wood",
    "Trial of Thunder",
];

const CONFIDENCE_THRESHOLD = 0.97;

async function main() {
    print("initializing tensorflow model...");
    const model = await tflite.loadTFLiteModel("botwqt.tflite");
    print("model loaded");
    console.log(model);

    async function getOutput(tensor: Tensor | Tensor[] | NamedTensorMap): Promise<[number, number]> {
        const output = tf.softmax((Array.isArray(tensor) ? tensor[0] : tensor) as Tensor);
        const data = await output.data();
        const max = Math.max(...data);
        if (max < CONFIDENCE_THRESHOLD) {
            return [0, max];
        }
        const index = data.indexOf(max);
        return [index, max];
    }

    async function predict(input: Uint8Array): Promise<Prediction> {
        print("preparing input tensor...");
        // expand 1 bit to 1 byte
        // const expanded = new Uint8Array(input.length * 8);
        // for (let i = 0; i < input.length; i++) {
        //     for (let j = 0; j < 8; j++) {
        //         expanded[i * 8 + j] = (input[i] >> j) & 1;
        //     }
        // }
        const expanded = input;
        for (let i = 0; i < expanded.length; i++) {
            if (expanded[i] !== 0 && expanded[i] !== 1) {
                print(`warning: invalid bit value: ${expanded[i]}, something wrong with the input?`);
                break;
            }
        }
        const inputTensor = tf.tensor4d(expanded, [1, 46, 492, 1]);
        print("running model...");
        const outputTensor = model.predict(inputTensor);
        print("preparing output...");
        const [questIdx, confidence] = await getOutput(outputTensor);
        print(`model result: ${questIdx}, confidence: ${Math.floor(confidence * 100)}%`);
        
        return {
            value: questIdx,
            quest: QUESTS[questIdx],
            confidence: confidence
        };
        
    }

    function errmsg(e: unknown): string {
        if (typeof e === "string") {
            return e;
        }
        if (e && typeof e === "object") {
            if ("message" in e) {
                return `${e.message}`;
            }
            if ("toString" in e && typeof e.toString === "function") {
                return e.toString();
            }
        }
        return `${e}`;
    }

    self.onmessage = async ({data}) => {
        const [type, value] = data;
        switch (type) {
            case "hostready":
                if (pingHost !== undefined) {
                    print("worker ready");
                }
                clearInterval(pingHost);
                break;
            case "predict":
                try {
                    const result = await predict(value);
                    //print(`predicted: ${result.value}`);
                    window.postMessage(["prediction", result]);
                } catch (e) {
                    console.error(e);
                    window.postMessage(["error", errmsg(e)]);
                }
                break;
        }
    };

    pingHost = setInterval(() => {
        self.postMessage(["ready"]);
    }, 100) as unknown as number;

    

}

main();
