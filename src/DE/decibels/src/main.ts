import { Application } from "./application.js";

export function main(argv: string[]): Promise<number> {
  const app = new Application();
  // @ts-expect-error gi.ts can't generate this, but it exists.
  return app.run(argv);
}
