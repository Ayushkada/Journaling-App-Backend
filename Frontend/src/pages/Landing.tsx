import { Link } from "react-router-dom";
import { BookOpen, Target, Brain } from "lucide-react";
import { useIsMobile } from "@/hooks/useMobile";

const Landing = () => {
  const isMobile = useIsMobile();

  if (isMobile) {
    return (
      <div className="min-h-screen flex items-center justify-center text-center px-4">
        <p className="text-lg text-white bg-black/70 rounded-lg p-6">
          📱 Mobile login view is coming soon. Please use a desktop for now.
        </p>
      </div>
    );
  }

  const HERO_BANNER_URL = "/hero-bg1.jpg";

  const features = [
    {
      icon: BookOpen,
      title: "Journaling",
      description:
        "Express your thoughts and emotions in a safe, private space designed for mindful reflection.",
    },
    {
      icon: Target,
      title: "Goal Tracking",
      description:
        "Set meaningful goals and track your progress with AI-powered insights and recommendations.",
    },
    {
      icon: Brain,
      title: "Mood & Self-Talk Analysis",
      description:
        "Understand your emotional patterns and inner dialogue through advanced sentiment analysis.",
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div
        className="relative h-screen flex items-center justify-center bg-cover bg-center"
        style={{
          backgroundImage: `linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url(${HERO_BANNER_URL})`,
        }}
      >
        <div className="text-center text-white px-4 max-w-4xl mx-auto">
          <h1 className="text-6xl font-bold mb-6 leading-tight">
            Your Journey to Mindful Living
          </h1>
          <p className="text-2xl mb-10 font-light text-white/90">
            "Start where you are. Use what you have. Do what you can."
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/auth"
              className="bg-primary text-white hover:primary px-8 py-4 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 shadow-md text-center"
            >
              Begin Your Journey
            </Link>
            <a
              href="#features"
              className="bg-transparent text-white border border-white hover:bg-white/10 px-8 py-4 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 text-center"
            >
              Explore Features
            </a>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div
        id="features"
        className="py-24 bg-gradient-to-b from-background to-accent/20"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-foreground mb-4">
              Everything You Need for Personal Growth
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Discover the power of mindful journaling combined with intelligent
              insights
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-20">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="bg-card p-8 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-2 border"
                >
                  <div className="flex items-center justify-center w-16 h-16 bg-primary/10 rounded-lg mb-6 mx-auto">
                    <Icon className="w-8 h-8 text-primary" />
                  </div>
                  <h3 className="text-2xl font-semibold mb-4 text-center text-card-foreground">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground text-center leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
          {/* CTA Section */}
          <div className="text-center bg-primary/5 py-16 px-8 rounded-2xl">
            <h3 className="text-3xl font-bold mb-4 text-foreground">
              Ready to Start Your Mindful Journey?
            </h3>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of people who have transformed their lives through
              mindful journaling and self-reflection.
            </p>
            <Link
              to="/auth"
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-10 py-4 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 inline-block shadow-lg"
            >
              Start Today
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;
