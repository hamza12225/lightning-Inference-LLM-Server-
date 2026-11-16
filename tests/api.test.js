import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

describe("AdaptiveGuard Anomaly API Documentation", () => {
  it("should open Swagger UI and inspect API schemas", async (context) => {
    const testdriver = TestDriver(context);

    await testdriver.provision.chrome({
      url: process.env.APP_URL || "https://herein-editorial-representations-viewed.trycloudflare.com/docs",
    });

    const pageTitle = await testdriver.assert("the Swagger UI documentation for PFE Anomaly API is displayed");
    expect(pageTitle).toBeTruthy();

    await testdriver.scroll("down", { amount: 600 });

    const schemaExpandArrow = await testdriver.find("expand arrow next to AnomalyRequest");
    await schemaExpandArrow.click();

    const assertResult = await testdriver.assert("the AnomalyRequest schema displays record and issues properties");
    expect(assertResult).toBeTruthy();
  });
});
