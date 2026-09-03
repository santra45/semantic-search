import { Card, Empty } from "../components/Bits";
import { useFilters } from "../lib/filters";

/**
 * Temporary. Each of these is replaced by a real screen; until then it proves
 * routing, the layout and the global filters are wired, rather than pretending
 * to show data.
 */
export function Placeholder({ title }: { title: string }) {
  const { environment, days } = useFilters();
  return (
    <Card title={title}>
      <Empty>
        Not built yet. Filters are live: environment <strong>{environment}</strong>,
        window <strong>{days} days</strong>.
      </Empty>
    </Card>
  );
}
