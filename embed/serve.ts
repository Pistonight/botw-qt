
const PORT = 3000;
console.log("Serve on http://localhost:" + PORT);


Bun.serve({
    port: PORT,
    fetch: async (req) => {
        const host = req.headers.get("host");
        const url = req.url;
        const [_, rawPath] = url.split(host);
        const path = rawPath === "/" ? "/index.html" : rawPath;
        console.log(`${rawPath} => dist${path}`)
        return new Response(Bun.file(`dist${path}`));
    }
});
