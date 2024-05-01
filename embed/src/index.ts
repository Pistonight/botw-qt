const uploadElem = document.getElementById("upload") as HTMLInputElement;
const canvasElem = document.getElementById("canvas") as HTMLCanvasElement;
const outputElem = document.getElementById("output") as HTMLElement;

const WIDTH = 492;
const HEIGHT = 46;

canvasElem.width = WIDTH;
canvasElem.height = HEIGHT;

declare const tf: typeof import ("@tensorflow/tfjs-core");

function print(text: string) {
    outputElem.innerText = text;
    console.log(text);
}

async function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    print("initializing worker...");
    const frame = document.getElementById("worker") as HTMLIFrameElement;
    while (!frame.contentWindow) {
        await sleep(1000);
    }
    const worker = frame.contentWindow;
    print("waiting for worker to be ready...");

    worker.addEventListener("message", async ({data}) => {
        const [type, value] = data;
        switch (type) {
            case "ready":
                if (uploadElem.disabled) {
                    print("worker initialized");
                    uploadElem.disabled = false;
                    worker.postMessage(["hostready"]);
                }
                break;
            case "error":
                print(`error: ${value}`);
                break;
            case "prediction":
                print(`predicted: ${value.quest}`)
                break;
        }
    });

    // convert png data into bitvec
    // the bit order is column major order, each byte is LSB first
    // i.e. the first byte is the first 8 pixels of the first column
    // the least significant bit is first pixel of the first column
    async function processInput(b64png: string): Promise<Uint8Array> {
        const img = new Image();
        await new Promise((resolve) => {
            img.onload = resolve;
            img.src = b64png;
        });
        img.width = WIDTH;
        img.height = HEIGHT;

        const context = canvasElem.getContext("2d");
        
        if (!context) {
            throw new Error("cannot get canvas context");
        }
        context.drawImage(img, 0, 0);
        // data is RGBA, row-major order
        const data = context.getImageData(0, 0, img.width, img.height);
        const output = new Uint8Array(WIDTH * HEIGHT);
        for (let row = 0; row < HEIGHT; row++) {
            for (let col = 0; col < WIDTH; col++) {
                const i = (row * WIDTH + col) * 4;
                const [r, g, b] = data.data.slice(i, i + 4);
                const gray = (r + g + b) / 3;
                const bit = gray < 128 ? 0 : 1;
                output[row * WIDTH + col] = bit;

                // const bitIndex = row * WIDTH + col;
                // const byteIndex = bitIndex >> 3;
                // const bitOffset = bitIndex & 0b111;
                // const oldByte = output[byteIndex] || 0;
                // output[byteIndex] = oldByte | (bit << bitOffset);
            }
        }

        return new Uint8Array(output);
    }

    async function predict(data: Uint8Array) {
        print("running prediction...");
        worker.postMessage(["predict", data]);
    }

    uploadElem.addEventListener("change", async () => {
        const files = uploadElem.files;
        if (!files) {
            return;
        }
        const file = await files[0].arrayBuffer();
        const b64 = btoa(String.fromCharCode(...new Uint8Array(file)));
        const data = await processInput(`data:image/png;base64,${b64}`);

        await predict(data);
    });
}

main();
