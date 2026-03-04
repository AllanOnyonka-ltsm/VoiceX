import { useEffect, useRef, useLayoutEffect, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { ArrowRight, Shield, Zap, Database, Cloud, Activity, Lock, ChevronRight, MapPin } from 'lucide-react';
import './App.css';
import GeoDashboard from './components/GeoDashboard';

gsap.registerPlugin(ScrollTrigger);

function App() {
  const mainRef = useRef<HTMLDivElement>(null);
  const heroRef = useRef<HTMLDivElement>(null);
  const capabilitiesRef = useRef<HTMLDivElement>(null);
  const performanceRef = useRef<HTMLDivElement>(null);
  const privacyRef = useRef<HTMLDivElement>(null);
  const workflowRef = useRef<HTMLDivElement>(null);
  const integrationRef = useRef<HTMLDivElement>(null);
  const trustRef = useRef<HTMLDivElement>(null);
  const geoRef = useRef<HTMLDivElement>(null);
  const closingRef = useRef<HTMLDivElement>(null);
  
  const [showGeoDashboard, setShowGeoDashboard] = useState(false);

  // Hero auto-play entrance animation
  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const heroTl = gsap.timeline({ defaults: { ease: 'power3.out' } });
      
      heroTl.fromTo('.hero-media', 
        { x: '-40vw', opacity: 0, scale: 0.98 },
        { x: 0, opacity: 1, scale: 1, duration: 0.6 },
        0
      );
      
      heroTl.fromTo('.hero-content',
        { x: '40vw', opacity: 0 },
        { x: 0, opacity: 1, duration: 0.55 },
        0.08
      );
      
      heroTl.fromTo('.hero-eyebrow',
        { y: -12, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.35 },
        0.22
      );
      
      heroTl.fromTo('.hero-headline span',
        { y: 18, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.4, stagger: 0.03 },
        0.30
      );
      
      heroTl.fromTo('.hero-subtext',
        { y: 14, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.35, stagger: 0.05 },
        0.50
      );
    }, heroRef);

    return () => ctx.revert();
  }, []);

  // Scroll-driven animations for all sections
  useEffect(() => {
    const ctx = gsap.context(() => {
      // Section 1: Hero - EXIT only
      const heroScrollTl = gsap.timeline({
        scrollTrigger: {
          trigger: heroRef.current,
          start: 'top top',
          end: '+=130%',
          pin: true,
          scrub: 0.6,
          onLeaveBack: () => {
            gsap.set('.hero-media, .hero-content', { opacity: 1, x: 0, scale: 1 });
          }
        }
      });

      heroScrollTl.fromTo('.hero-media',
        { x: 0, opacity: 1, scale: 1 },
        { x: '-18vw', opacity: 0, scale: 0.98, ease: 'power2.in' },
        0.70
      );
      heroScrollTl.fromTo('.hero-content',
        { x: 0, opacity: 1 },
        { x: '18vw', opacity: 0, ease: 'power2.in' },
        0.70
      );

      // Section 2: Capabilities Mosaic
      const capabilitiesTl = gsap.timeline({
        scrollTrigger: {
          trigger: capabilitiesRef.current,
          start: 'top top',
          end: '+=130%',
          pin: true,
          scrub: 0.6,
        }
      });

      capabilitiesTl.fromTo('.cap-left',
        { x: '-60vw', opacity: 0, scale: 0.98 },
        { x: 0, opacity: 1, scale: 1, ease: 'none' },
        0
      );
      capabilitiesTl.fromTo('.cap-top-right',
        { x: '60vw', opacity: 0 },
        { x: 0, opacity: 1, ease: 'none' },
        0.06
      );
      capabilitiesTl.fromTo('.cap-bottom-right',
        { y: '60vh', opacity: 0, scale: 0.98 },
        { y: 0, opacity: 1, scale: 1, ease: 'none' },
        0.10
      );
      capabilitiesTl.fromTo('.cap-text-content',
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.02, ease: 'none' },
        0.14
      );

      capabilitiesTl.fromTo('.cap-left',
        { x: 0, opacity: 1 },
        { x: '-14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );
      capabilitiesTl.fromTo('.cap-top-right',
        { x: 0, opacity: 1 },
        { x: '14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );
      capabilitiesTl.fromTo('.cap-bottom-right',
        { y: 0, opacity: 1 },
        { y: '14vh', opacity: 0, ease: 'power2.in' },
        0.70
      );

      // Section 3: Performance
      const performanceTl = gsap.timeline({
        scrollTrigger: {
          trigger: performanceRef.current,
          start: 'top top',
          end: '+=130%',
          pin: true,
          scrub: 0.6,
        }
      });

      performanceTl.fromTo('.perf-left',
        { x: '-70vw', opacity: 0 },
        { x: 0, opacity: 1, ease: 'none' },
        0
      );
      performanceTl.fromTo('.perf-right',
        { x: '70vw', opacity: 0, scale: 0.98 },
        { x: 0, opacity: 1, scale: 1, ease: 'none' },
        0.06
      );
      performanceTl.fromTo('.perf-text',
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.02, ease: 'none' },
        0.10
      );

      performanceTl.fromTo('.perf-left',
        { x: 0, opacity: 1 },
        { x: '-14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );
      performanceTl.fromTo('.perf-right',
        { x: 0, opacity: 1 },
        { x: '14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );

      // Section 4: Privacy (reversed)
      const privacyTl = gsap.timeline({
        scrollTrigger: {
          trigger: privacyRef.current,
          start: 'top top',
          end: '+=130%',
          pin: true,
          scrub: 0.6,
        }
      });

      privacyTl.fromTo('.priv-left',
        { x: '-70vw', opacity: 0, scale: 0.98 },
        { x: 0, opacity: 1, scale: 1, ease: 'none' },
        0
      );
      privacyTl.fromTo('.priv-right',
        { x: '70vw', opacity: 0 },
        { x: 0, opacity: 1, ease: 'none' },
        0.06
      );
      privacyTl.fromTo('.priv-text',
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.02, ease: 'none' },
        0.10
      );

      privacyTl.fromTo('.priv-left',
        { x: 0, opacity: 1 },
        { x: '-14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );
      privacyTl.fromTo('.priv-right',
        { x: 0, opacity: 1 },
        { x: '14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );

      // Section 5: Workflow
      const workflowTl = gsap.timeline({
        scrollTrigger: {
          trigger: workflowRef.current,
          start: 'top top',
          end: '+=130%',
          pin: true,
          scrub: 0.6,
        }
      });

      workflowTl.fromTo('.work-left',
        { x: '-70vw', opacity: 0 },
        { x: 0, opacity: 1, ease: 'none' },
        0
      );
      workflowTl.fromTo('.work-right',
        { x: '70vw', opacity: 0, scale: 0.98 },
        { x: 0, opacity: 1, scale: 1, ease: 'none' },
        0.06
      );
      workflowTl.fromTo('.work-text',
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.02, ease: 'none' },
        0.10
      );
      workflowTl.fromTo('.work-steps',
        { scale: 0.8, opacity: 0 },
        { scale: 1, opacity: 1, stagger: 0.03, ease: 'none' },
        0.14
      );

      workflowTl.fromTo('.work-left',
        { x: 0, opacity: 1 },
        { x: '-14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );
      workflowTl.fromTo('.work-right',
        { x: 0, opacity: 1 },
        { x: '14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );

      // Section 6: Integration (reversed)
      const integrationTl = gsap.timeline({
        scrollTrigger: {
          trigger: integrationRef.current,
          start: 'top top',
          end: '+=130%',
          pin: true,
          scrub: 0.6,
        }
      });

      integrationTl.fromTo('.int-left',
        { x: '-70vw', opacity: 0, scale: 0.98 },
        { x: 0, opacity: 1, scale: 1, ease: 'none' },
        0
      );
      integrationTl.fromTo('.int-right',
        { x: '70vw', opacity: 0 },
        { x: 0, opacity: 1, ease: 'none' },
        0.06
      );
      integrationTl.fromTo('.int-text',
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.02, ease: 'none' },
        0.10
      );
      integrationTl.fromTo('.int-chips',
        { y: 10, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.03, ease: 'none' },
        0.16
      );

      integrationTl.fromTo('.int-left',
        { x: 0, opacity: 1 },
        { x: '-14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );
      integrationTl.fromTo('.int-right',
        { x: 0, opacity: 1 },
        { x: '14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );

      // Section 7: Trust
      const trustTl = gsap.timeline({
        scrollTrigger: {
          trigger: trustRef.current,
          start: 'top top',
          end: '+=130%',
          pin: true,
          scrub: 0.6,
        }
      });

      trustTl.fromTo('.trust-left',
        { x: '-70vw', opacity: 0 },
        { x: 0, opacity: 1, ease: 'none' },
        0
      );
      trustTl.fromTo('.trust-right',
        { x: '70vw', opacity: 0, scale: 0.98 },
        { x: 0, opacity: 1, scale: 1, ease: 'none' },
        0.06
      );
      trustTl.fromTo('.trust-text',
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.02, ease: 'none' },
        0.10
      );

      trustTl.fromTo('.trust-left',
        { x: 0, opacity: 1 },
        { x: '-14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );
      trustTl.fromTo('.trust-right',
        { x: 0, opacity: 1 },
        { x: '14vw', opacity: 0, ease: 'power2.in' },
        0.70
      );

      // Section 8: Geographic Dashboard (flowing, not pinned)
      gsap.fromTo('.geo-content',
        { y: '6vh', opacity: 0, scale: 0.98 },
        {
          y: 0, opacity: 1, scale: 1,
          scrollTrigger: {
            trigger: geoRef.current,
            start: 'top 80%',
            end: 'top 55%',
            scrub: 0.5,
          }
        }
      );

      // Section 9: Closing CTA
      gsap.fromTo('.closing-card',
        { y: '6vh', opacity: 0, scale: 0.98 },
        {
          y: 0, opacity: 1, scale: 1,
          scrollTrigger: {
            trigger: closingRef.current,
            start: 'top 80%',
            end: 'top 55%',
            scrub: 0.5,
          }
        }
      );

      gsap.fromTo('.closing-text',
        { y: 14, opacity: 0 },
        {
          y: 0, opacity: 1, stagger: 0.03,
          scrollTrigger: {
            trigger: closingRef.current,
            start: 'top 70%',
            end: 'top 50%',
            scrub: 0.5,
          }
        }
      );

      // Global snap for pinned sections
      const pinned = ScrollTrigger.getAll()
        .filter(st => st.vars.pin)
        .sort((a, b) => a.start - b.start);
      
      const maxScroll = ScrollTrigger.maxScroll(window);
      if (maxScroll && pinned.length > 0) {
        const pinnedRanges = pinned.map(st => ({
          start: st.start / maxScroll,
          end: (st.end ?? st.start) / maxScroll,
          center: (st.start + ((st.end ?? st.start) - st.start) * 0.5) / maxScroll,
        }));

        ScrollTrigger.create({
          snap: {
            snapTo: (value: number) => {
              const inPinned = pinnedRanges.some(r => value >= r.start - 0.02 && value <= r.end + 0.02);
              if (!inPinned) return value;
              
              const target = pinnedRanges.reduce((closest, r) =>
                Math.abs(r.center - value) < Math.abs(closest - value) ? r.center : closest,
                pinnedRanges[0]?.center ?? 0
              );
              return target;
            },
            duration: { min: 0.15, max: 0.35 },
            delay: 0,
            ease: 'power2.out',
          }
        });
      }
    }, mainRef);

    return () => ctx.revert();
  }, []);

  if (showGeoDashboard) {
    return <GeoDashboard />;
  }

  return (
    <div ref={mainRef} className="relative">
      {/* Grain overlay */}
      <div className="grain-overlay" />

      {/* Persistent Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-[100] px-[6vw] py-5 flex items-center justify-between bg-[#F6F7F9]/80 backdrop-blur-sm">
        <div className="text-xl font-bold text-[#0B0D10]" style={{ fontFamily: 'Sora, sans-serif' }}>
          VoiceX
        </div>
        <div className="hidden md:flex items-center gap-8">
          <a href="#product" className="text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors">Product</a>
          <a href="#capabilities" className="text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors">Capabilities</a>
          <a href="#security" className="text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors">Security</a>
          <a href="#geographic" className="text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1">
            <MapPin className="w-3 h-3" />
            Map
          </a>
          <a href="#contact" className="text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors">Contact</a>
        </div>
        <button className="px-5 py-2 border border-[#11A300] text-[#11A300] rounded-full text-sm font-medium hover:bg-[#11A300] hover:text-white transition-all">
          Request access
        </button>
      </nav>

      {/* Section 1: Hero */}
      <section ref={heroRef} className="section-pinned bg-[#F6F7F9] z-10">
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <div className="hero-media absolute left-[6vw] top-[14vh] w-[54vw] h-[72vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/hero_labs.jpg" 
              alt="Clinical laboratory" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
          
          <div className="hero-content absolute left-[62vw] top-[14vh] w-[32vw] h-[72vh] bg-white rounded-[28px] card-shadow p-[clamp(22px,2.2vw,36px)] flex flex-col justify-between">
            <div>
              <span className="hero-eyebrow pill pill-green-outline mb-6">Offline-first</span>
              <h1 className="hero-headline heading-lg text-[#0B0D10] mt-6">
                <span className="block">AI audio triage</span>
                <span className="block">for TB risk &</span>
                <span className="block">voice pathology</span>
              </h1>
            </div>
            
            <div>
              <p className="hero-subtext text-[#6B7280] text-base leading-relaxed mb-8">
                Record, analyze, and triage on-device—no connectivity required.
              </p>
              
              <div className="hero-subtext flex items-center gap-4 mb-8">
                <button className="px-6 py-3 bg-[#11A300] text-white rounded-full text-sm font-medium hover:bg-[#0e8a00] transition-colors flex items-center gap-2">
                  Request access
                  <ArrowRight className="w-4 h-4" />
                </button>
                <a href="#capabilities" className="text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1">
                  View capabilities
                  <ChevronRight className="w-4 h-4" />
                </a>
              </div>
              
              <p className="hero-subtext text-xs text-[#6B7280]">
                Built for low-resource settings. HIPAA-aligned. Field-ready.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: Capabilities Mosaic */}
      <section ref={capabilitiesRef} id="capabilities" className="section-pinned bg-[#F6F7F9] z-20">
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <div className="cap-left absolute left-[6vw] top-[14vh] w-[34vw] h-[72vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/capabilities_microphone.jpg" 
              alt="Professional microphone" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
          
          <div className="cap-top-right absolute left-[42vw] top-[14vh] w-[52vw] h-[34vh] bg-white rounded-[28px] card-shadow p-8 flex flex-col justify-between">
            <div>
              <span className="cap-text-content pill pill-green mb-4">Capabilities</span>
              <h2 className="cap-text-content heading-md text-[#0B0D10] mt-4">Three ways to listen.</h2>
              <p className="cap-text-content text-[#6B7280] mt-4 max-w-lg">
                Cough detection, sustained vowel analysis, and phrase reading—each optimized for clinical signal and low-noise capture.
              </p>
            </div>
            <a href="#" className="cap-text-content text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1 self-end">
              Explore the pipeline
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
          
          <div className="cap-bottom-right absolute left-[42vw] top-[52vh] w-[52vw] h-[34vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/capabilities_interface.jpg" 
              alt="App interface" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
        </div>
      </section>

      {/* Section 3: Performance */}
      <section ref={performanceRef} className="section-pinned bg-[#F6F7F9] z-30">
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <div className="perf-left absolute left-[6vw] top-[14vh] w-[38vw] h-[72vh] bg-white rounded-[28px] card-shadow p-8 flex flex-col justify-between">
            <div>
              <span className="perf-text pill pill-green-outline mb-4">Speed</span>
              <h2 className="perf-text heading-md text-[#0B0D10] mt-4">Real-time inference.</h2>
              <p className="perf-text text-[#6B7280] mt-4">
                TensorFlow Lite and ONNX Runtime deliver sub‑5‑second results on low‑end Android—no cloud needed.
              </p>
              
              <ul className="perf-text mt-8 space-y-4">
                <li className="flex items-start gap-3">
                  <Zap className="w-5 h-5 text-[#11A300] mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-[#0B0D10]">INT8 quantization for tiny footprint</span>
                </li>
                <li className="flex items-start gap-3">
                  <Activity className="w-5 h-5 text-[#11A300] mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-[#0B0D10]">Energy-aware scheduling</span>
                </li>
                <li className="flex items-start gap-3">
                  <Database className="w-5 h-5 text-[#11A300] mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-[#0B0D10]">Battery & thermal safe</span>
                </li>
              </ul>
            </div>
            <a href="#" className="perf-text text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1">
              See benchmarks
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
          
          <div className="perf-right absolute left-[46vw] top-[14vh] w-[48vw] h-[72vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/performance_device.jpg" 
              alt="Healthcare worker with device" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
        </div>
      </section>

      {/* Section 4: Privacy */}
      <section ref={privacyRef} id="security" className="section-pinned bg-[#F6F7F9] z-40">
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <div className="priv-left absolute left-[6vw] top-[14vh] w-[48vw] h-[72vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/privacy_lock.jpg" 
              alt="Security lock" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
          
          <div className="priv-right absolute left-[56vw] top-[14vh] w-[38vw] h-[72vh] bg-white rounded-[28px] card-shadow p-8 flex flex-col justify-between">
            <div>
              <span className="priv-text pill pill-green mb-4">Privacy</span>
              <h2 className="priv-text heading-md text-[#0B0D10] mt-4">Patient data stays local.</h2>
              <p className="priv-text text-[#6B7280] mt-4">
                Recordings and metadata are stored on-device with encryption. Sync only when you choose—and only anonymized payloads.
              </p>
            </div>
            
            <div>
              <div className="priv-text flex flex-wrap gap-2 mb-6">
                <span className="px-3 py-1.5 bg-[#F6F7F9] rounded-full text-xs font-medium text-[#0B0D10]">SQLite + JSON local store</span>
                <span className="px-3 py-1.5 bg-[#F6F7F9] rounded-full text-xs font-medium text-[#0B0D10]">AES-256 at rest</span>
                <span className="px-3 py-1.5 bg-[#F6F7F9] rounded-full text-xs font-medium text-[#0B0D10]">TLS 1.3 in transit</span>
              </div>
              <a href="#" className="priv-text text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1">
                Security overview
                <ArrowRight className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Section 5: Workflow */}
      <section ref={workflowRef} className="section-pinned bg-[#F6F7F9] z-50">
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <div className="work-left absolute left-[6vw] top-[14vh] w-[38vw] h-[72vh] bg-white rounded-[28px] card-shadow p-8 flex flex-col justify-between">
            <div>
              <span className="work-text pill pill-green-outline mb-4">Workflow</span>
              <h2 className="work-text heading-md text-[#0B0D10] mt-4">Guided capture.</h2>
              <p className="work-text text-[#6B7280] mt-4">
                On-screen prompts help patients produce clean, repeatable audio—reducing retakes and improving model confidence.
              </p>
              
              <div className="work-steps mt-8 space-y-4">
                <div className="flex items-center gap-4">
                  <span className="w-8 h-8 rounded-full bg-[#11A300] text-white flex items-center justify-center text-sm font-bold">01</span>
                  <span className="text-sm text-[#0B0D10]">Position the device</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="w-8 h-8 rounded-full bg-[#11A300] text-white flex items-center justify-center text-sm font-bold">02</span>
                  <span className="text-sm text-[#0B0D10]">Follow the prompt</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="w-8 h-8 rounded-full bg-[#11A300] text-white flex items-center justify-center text-sm font-bold">03</span>
                  <span className="text-sm text-[#0B0D10]">Review & confirm</span>
                </div>
              </div>
            </div>
            <a href="#" className="work-text text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1">
              Watch a demo
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
          
          <div className="work-right absolute left-[46vw] top-[14vh] w-[48vw] h-[72vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/workflow_guided.jpg" 
              alt="Guided patient interaction" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
        </div>
      </section>

      {/* Section 6: Integration */}
      <section ref={integrationRef} className="section-pinned bg-[#F6F7F9] z-[60]">
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <div className="int-left absolute left-[6vw] top-[14vh] w-[48vw] h-[72vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/integration_cloud.jpg" 
              alt="Cloud integration" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
          
          <div className="int-right absolute left-[56vw] top-[14vh] w-[38vw] h-[72vh] bg-white rounded-[28px] card-shadow p-8 flex flex-col justify-between">
            <div>
              <span className="int-text pill pill-green mb-4">Integration</span>
              <h2 className="int-text heading-md text-[#0B0D10] mt-4">Cloud sync when you need it.</h2>
              <p className="int-text text-[#6B7280] mt-4">
                Enable secure upload for aggregation, retraining, and reporting. On-device, get explainability and clinical metrics clinicians can act on.
              </p>
              
              <div className="int-chips flex flex-wrap gap-2 mt-6">
                <span className="px-3 py-1.5 bg-[#F6F7F9] rounded-full text-xs font-medium text-[#0B0D10] flex items-center gap-1">
                  <Activity className="w-3 h-3 text-[#11A300]" />
                  SHAP + Grad‑CAM++
                </span>
                <span className="px-3 py-1.5 bg-[#F6F7F9] rounded-full text-xs font-medium text-[#0B0D10] flex items-center gap-1">
                  <Zap className="w-3 h-3 text-[#11A300]" />
                  Jitter / Shimmer / HNR
                </span>
              </div>
            </div>
            <a href="#" className="int-text text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1">
              Read the docs
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </div>
      </section>

      {/* Section 7: Trust */}
      <section ref={trustRef} className="section-pinned bg-[#F6F7F9] z-[70]">
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <div className="trust-left absolute left-[6vw] top-[14vh] w-[38vw] h-[72vh] bg-white rounded-[28px] card-shadow p-8 flex flex-col justify-between">
            <div>
              <span className="trust-text pill pill-green-outline mb-4">Trust</span>
              <h2 className="trust-text heading-md text-[#0B0D10] mt-4">Built for real‑world conditions.</h2>
              <p className="trust-text text-[#6B7280] mt-4">
                OTA model updates, robust error handling, and low-resource tuning—so the system keeps working when infrastructure doesn't.
              </p>
              
              <ul className="trust-text mt-8 space-y-4">
                <li className="flex items-start gap-3">
                  <Cloud className="w-5 h-5 text-[#11A300] mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-[#0B0D10]">Model versioning + rollback</span>
                </li>
                <li className="flex items-start gap-3">
                  <Database className="w-5 h-5 text-[#11A300] mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-[#0B0D10]">Offline logging & diagnostics</span>
                </li>
                <li className="flex items-start gap-3">
                  <Lock className="w-5 h-5 text-[#11A300] mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-[#0B0D10]">Conflict resolution on sync</span>
                </li>
              </ul>
            </div>
            <a href="#" className="trust-text text-sm font-medium text-[#0B0D10] hover:text-[#11A300] transition-colors flex items-center gap-1">
              Deployment guide
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
          
          <div className="trust-right absolute left-[46vw] top-[14vh] w-[48vw] h-[72vh] rounded-[28px] overflow-hidden card-shadow">
            <img 
              src="/trust_field.jpg" 
              alt="Field-ready device" 
              className="w-full h-full object-cover slow-zoom"
            />
          </div>
        </div>
      </section>

      {/* Section 8: Geographic Dashboard */}
      <section ref={geoRef} id="geographic" className="relative bg-[#F6F7F9] min-h-screen z-[75] py-20">
        <div className="geo-content px-[6vw]">
          <div className="text-center mb-12">
            <span className="pill pill-green mb-4 inline-block">Geographic Surveillance</span>
            <h2 className="heading-lg text-[#0B0D10] mt-4">Mapping Disease Clusters</h2>
            <p className="text-[#6B7280] mt-4 max-w-2xl mx-auto">
              Real-time geographic tracking of TB risk clusters and voice pathology detections. 
              Identify hotspots, track trends, and deploy resources effectively.
            </p>
          </div>

          {/* Preview Card */}
          <div className="bg-white rounded-[28px] overflow-hidden card-shadow max-w-5xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2">
              <div className="p-8 flex flex-col justify-center">
                <div className="flex items-center gap-3 mb-4">
                  <MapPin className="w-6 h-6 text-[#11A300]" />
                  <h3 className="text-xl font-bold text-[#0B0D10]">Live Detection Map</h3>
                </div>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-3">
                    <Activity className="w-5 h-5 text-[#11A300] mt-0.5" />
                    <span className="text-sm text-[#6B7280]">DBSCAN clustering for hotspot detection</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Zap className="w-5 h-5 text-[#11A300] mt-0.5" />
                    <span className="text-sm text-[#6B7280]">Heatmap visualization with risk-weighted intensity</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Database className="w-5 h-5 text-[#11A300] mt-0.5" />
                    <span className="text-sm text-[#6B7280]">Time-series trend analysis</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Cloud className="w-5 h-5 text-[#11A300] mt-0.5" />
                    <span className="text-sm text-[#6B7280]">Anonymized data aggregation for research</span>
                  </li>
                </ul>
                <button 
                  onClick={() => setShowGeoDashboard(true)}
                  className="px-6 py-3 bg-[#11A300] text-white rounded-full text-sm font-medium hover:bg-[#0e8a00] transition-colors flex items-center gap-2 w-fit"
                >
                  Open Dashboard
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
              <div className="bg-gradient-to-br from-[#11A300]/10 to-[#11A300]/5 p-8 flex items-center justify-center">
                <div className="relative">
                  {/* Mock Map Visualization */}
                  <div className="w-64 h-64 rounded-2xl bg-white shadow-lg relative overflow-hidden">
                    <div className="absolute inset-0 bg-[#e5e7eb]">
                      {/* Mock grid */}
                      <div className="absolute inset-0" style={{
                        backgroundImage: 'linear-gradient(#d1d5db 1px, transparent 1px), linear-gradient(90deg, #d1d5db 1px, transparent 1px)',
                        backgroundSize: '20px 20px'
                      }} />
                      {/* Mock clusters */}
                      <div className="absolute top-1/4 left-1/4 w-12 h-12 rounded-full bg-[#dc2626]/40 animate-pulse" />
                      <div className="absolute top-1/2 right-1/3 w-8 h-8 rounded-full bg-[#ea580c]/40" />
                      <div className="absolute bottom-1/3 left-1/2 w-10 h-10 rounded-full bg-[#ca8a04]/40" />
                      <div className="absolute top-1/3 right-1/4 w-6 h-6 rounded-full bg-[#16a34a]/40" />
                    </div>
                  </div>
                  {/* Stats overlay */}
                  <div className="absolute -bottom-4 -right-4 bg-white rounded-xl p-3 shadow-lg">
                    <p className="text-xs text-[#6B7280]">Active Clusters</p>
                    <p className="text-xl font-bold text-[#0B0D10]">8</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8 max-w-5xl mx-auto">
            <div className="bg-white rounded-xl p-4 card-shadow text-center">
              <p className="text-2xl font-bold text-[#0B0D10]">2,164</p>
              <p className="text-xs text-[#6B7280]">Total Detections</p>
            </div>
            <div className="bg-white rounded-xl p-4 card-shadow text-center">
              <p className="text-2xl font-bold text-[#dc2626]">342</p>
              <p className="text-xs text-[#6B7280]">High Risk Cases</p>
            </div>
            <div className="bg-white rounded-xl p-4 card-shadow text-center">
              <p className="text-2xl font-bold text-[#0B0D10]">25.4</p>
              <p className="text-xs text-[#6B7280]">Coverage km²</p>
            </div>
            <div className="bg-white rounded-xl p-4 card-shadow text-center">
              <p className="text-2xl font-bold text-[#11A300]">+23%</p>
              <p className="text-xs text-[#6B7280]">Monthly Growth</p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 9: Closing CTA + Footer */}
      <section ref={closingRef} id="contact" className="relative bg-[#0B0D10] min-h-screen z-[80]">
        <div className="px-[6vw] py-[10vh]">
          <div className="closing-card w-full min-h-[52vh] rounded-[28px] border border-white/10 p-8 md:p-12 flex flex-col justify-between">
            <div>
              <span className="closing-text pill pill-green mb-6">Get started</span>
              <h2 className="closing-text heading-lg text-white mt-6">Ready to deploy?</h2>
              <p className="closing-text text-white/60 mt-4 max-w-xl">
                Request early access, schedule a field pilot, or integrate VoiceX into your existing screening program.
              </p>
            </div>
            
            <div>
              <div className="closing-text flex flex-wrap gap-4 mb-8">
                <button className="px-6 py-3 bg-[#11A300] text-white rounded-full text-sm font-medium hover:bg-[#0e8a00] transition-colors flex items-center gap-2">
                  Request access
                  <ArrowRight className="w-4 h-4" />
                </button>
                <button className="px-6 py-3 border border-white/30 text-white rounded-full text-sm font-medium hover:bg-white/10 transition-colors">
                  Contact sales
                </button>
              </div>
              <p className="closing-text text-xs text-white/40 flex items-center gap-2">
                <Shield className="w-3 h-3" />
                HIPAA-aligned • Field-tested • Offline-first
              </p>
            </div>
          </div>
          
          <footer className="mt-16 pt-8 border-t border-white/10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <div>
                <h4 className="text-white font-semibold mb-4">Product</h4>
                <ul className="space-y-2">
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Capabilities</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Performance</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Security</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Roadmap</a></li>
                </ul>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-4">Resources</h4>
                <ul className="space-y-2">
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Docs</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">API</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Benchmarks</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Deployment guide</a></li>
                </ul>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-4">Company</h4>
                <ul className="space-y-2">
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">About</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Contact</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Privacy</a></li>
                  <li><a href="#" className="text-white/60 text-sm hover:text-white transition-colors">Terms</a></li>
                </ul>
              </div>
              <div className="flex items-end">
                <p className="text-white/40 text-sm">© VoiceX Health Technologies</p>
              </div>
            </div>
          </footer>
        </div>
      </section>
    </div>
  );
}

export default App;
