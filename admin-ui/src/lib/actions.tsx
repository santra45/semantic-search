import { useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "./toast";
import { ConfirmSheet, type ConfirmConfig } from "../components/ConfirmSheet";

/**
 * Opening a confirm sheet, running the write, and telling the operator what
 * happened — in one place, so every action behaves the same way.
 *
 * WHAT "UNDO" MEANS HERE, precisely. §7.4 wants a one-click revert, and the
 * honest implementation is calling the INVERSE endpoint — disable → enable —
 * not replaying an audit row. Two consequences worth being straight about:
 *
 *   · It writes a second audit entry. The trail shows a disable and an enable,
 *     which is what actually happened, rather than pretending the first never
 *     did.
 *   · Some actions have no inverse. A rotation DELETES the key it replaces, so
 *     there is nothing to put back; those simply do not offer undo rather than
 *     offering one that quietly fails.
 */

export interface ActionResult {
  message: string;
  detail?: string;
  /** Omit when the action cannot be undone. */
  undo?: { label: string; run: () => Promise<unknown> };
}

export function useActions(invalidateKeys: unknown[][]) {
  const qc = useQueryClient();
  const toast = useToast();
  const [sheet, setSheet] = useState<ConfirmConfig | null>(null);

  const refresh = () => {
    for (const key of invalidateKeys) qc.invalidateQueries({ queryKey: key });
  };

  /** Describe the action, and what to do when it succeeds. */
  function open(
    config: Omit<ConfirmConfig, "onConfirm">,
    perform: (reason: string, values: Record<string, string>) => Promise<ActionResult>,
  ) {
    setSheet({
      ...config,
      onConfirm: async (reason, values) => {
        const result = await perform(reason, values);
        refresh();
        toast.push({
          kind: "success",
          message: result.message,
          detail: result.detail,
          action: result.undo
            ? {
                label: result.undo.label,
                run: () => {
                  result
                    .undo!.run()
                    .then(() => {
                      refresh();
                      toast.success("Reverted", "A second audit entry records it.");
                    })
                    .catch((e) =>
                      toast.error("Could not revert", e instanceof Error ? e.message : String(e)),
                    );
                },
              }
            : undefined,
        });
      },
    });
  }

  const sheetNode: ReactNode = sheet ? (
    <ConfirmSheet config={sheet} onClose={() => setSheet(null)} />
  ) : null;

  return { open, sheetNode, toast, refresh };
}
