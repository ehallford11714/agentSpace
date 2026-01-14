const features = [
  {
    title: "Neural Orchestration",
    copy: "Coordinate multi-agent workflows with adaptive routing and memory layers.",
  },
  {
    title: "Realtime Observability",
    copy: "Visualize traces, costs, and throughput with precision dashboards.",
  },
  {
    title: "Secure Automation",
    copy: "Ship automation with guardrails, policy checks, and human-in-loop handoffs.",
  },
];

const metrics = [
  { label: "Latency", value: "120ms", helper: "p95 response" },
  { label: "Agents", value: "64+", helper: "active nodes" },
  { label: "Uptime", value: "99.98%", helper: "last 90 days" },
];

const steps = [
  {
    title: "Design the mission",
    copy: "Map objectives and define success signals for every agent squad.",
  },
  {
    title: "Deploy to the edge",
    copy: "Ship workloads to any cloud, region, or secure enclave.",
  },
  {
    title: "Evolve continuously",
    copy: "Tune prompts, policies, and memory to stay ahead of change.",
  },
];

export default function App() {
  return (
    <div className="app">
      <header className="hero">
        <nav className="nav">
          <span className="logo">AgentSpace</span>
          <div className="nav-links">
            <a href="#platform">Platform</a>
            <a href="#signals">Signals</a>
            <a href="#edge">Edge</a>
            <button className="button ghost">Request access</button>
          </div>
        </nav>

        <div className="hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">Modern edge ops for agentic systems</p>
            <h1>
              Build autonomous workflows with a <span>bold, adaptive</span>
              <br />
              frontend that feels alive.
            </h1>
            <p className="lead">
              AgentSpace brings orchestration, observability, and governance into a
              single edge-ready workspace. Move faster with a UI designed for real-time
              intelligence.
            </p>
            <div className="cta-row">
              <button className="button primary">Launch console</button>
              <button className="button secondary">Explore docs</button>
            </div>
            <div className="hero-badges">
              <span>Edge-grade UI</span>
              <span>Zero-latency updates</span>
              <span>Composable flows</span>
            </div>
          </div>

          <div className="hero-panel" aria-label="Live system preview">
            <div className="panel-header">
              <div>
                <p className="panel-title">Live mission matrix</p>
                <p className="panel-sub">Status across distributed agents</p>
              </div>
              <button className="chip">Synced</button>
            </div>
            <div className="panel-body">
              <div className="signal-card">
                <p>Inference queue</p>
                <strong>12 active</strong>
                <span className="trend up">+18%</span>
              </div>
              <div className="signal-card">
                <p>Memory graph</p>
                <strong>824 nodes</strong>
                <span className="trend">Stable</span>
              </div>
              <div className="signal-card">
                <p>Policy checks</p>
                <strong>0 anomalies</strong>
                <span className="trend up">Running</span>
              </div>
            </div>
            <div className="panel-footer">
              <div className="pulse" />
              <p>Streaming updates from edge clusters</p>
            </div>
          </div>
        </div>
      </header>

      <section className="section" id="platform">
        <div className="section-header">
          <h2>Everything you need to orchestrate at the edge</h2>
          <p>
            A modern interface built for high velocity teams running mission-critical
            automations.
          </p>
        </div>
        <div className="feature-grid">
          {features.map((item) => (
            <article key={item.title} className="feature-card">
              <h3>{item.title}</h3>
              <p>{item.copy}</p>
              <button className="text-link">Learn more →</button>
            </article>
          ))}
        </div>
      </section>

      <section className="section split" id="signals">
        <div>
          <h2>Signal clarity in every second</h2>
          <p>
            Blend telemetry, agent performance, and guardrails into a single
            command view. Everything is designed to be glanceable and
            real-time.
          </p>
          <div className="metric-row">
            {metrics.map((metric) => (
              <div className="metric" key={metric.label}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
                <em>{metric.helper}</em>
              </div>
            ))}
          </div>
        </div>
        <div className="signal-stack">
          <div className="signal-line">
            <span>Pipeline velocity</span>
            <div className="line" />
          </div>
          <div className="signal-line">
            <span>Agent load</span>
            <div className="line" />
          </div>
          <div className="signal-line">
            <span>Memory sync</span>
            <div className="line" />
          </div>
        </div>
      </section>

      <section className="section" id="edge">
        <div className="section-header">
          <h2>Launch your edge journey in minutes</h2>
          <p>
            Accelerate from concept to production with a guided, modern workflow
            built for scale.
          </p>
        </div>
        <div className="timeline">
          {steps.map((step, index) => (
            <div className="timeline-step" key={step.title}>
              <span className="step-index">0{index + 1}</span>
              <div>
                <h3>{step.title}</h3>
                <p>{step.copy}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="section cta">
        <div>
          <h2>Ready to design your next-edge interface?</h2>
          <p>
            Join teams shipping agentic experiences with a UI that feels futuristic
            and stays reliable at scale.
          </p>
        </div>
        <div className="cta-actions">
          <button className="button primary">Book a demo</button>
          <button className="button ghost">View changelog</button>
        </div>
      </section>
    </div>
  );
}
