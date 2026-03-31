<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CityPulse</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Google Font -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap" rel="stylesheet">

  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Poppins', sans-serif;
    }

    body {
      overflow-x: hidden;
      background: #0a0f1c;
      color: white;
    }

    /* HERO SECTION */
    .hero {
      height: 200vh; /* extra space for scroll animation */
      position: relative;
      overflow: hidden;
    }

    /* BACKGROUND GLOW */
    .bg-glow {
      position: absolute;
      width: 600px;
      height: 600px;
      background: radial-gradient(circle, rgba(0,150,255,0.4), transparent);
      top: 20%;
      left: 50%;
      transform: translateX(-50%);
      filter: blur(120px);
      z-index: 0;
    }

    /* METRO TRAIN */
    .metro {
      position: fixed;
      top: 40%;
      left: -300px;
      width: 250px;
      z-index: 2;
      transition: transform 0.2s linear;
    }

    /* TITLE CONTAINER */
    .content {
      position: sticky;
      top: 30%;
      text-align: center;
      z-index: 3;
      padding: 20px;
    }

    .title {
      font-size: 4rem;
      font-weight: 800;
      background: linear-gradient(90deg, #00c6ff, #0072ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      opacity: 0;
      transform: translateY(50px);
      transition: all 1s ease;
    }

    .subtitle {
      margin-top: 20px;
      font-size: 1.2rem;
      max-width: 900px;
      margin-left: auto;
      margin-right: auto;
      color: #cfd8ff;
      opacity: 0;
      transform: translateY(50px);
      transition: all 1.2s ease;
    }

    .show {
      opacity: 1 !important;
      transform: translateY(0) !important;
    }

    /* GLASS BUTTON */
    .cta-btn {
      margin-top: 30px;
      padding: 14px 30px;
      border-radius: 30px;
      border: 1px solid rgba(255,255,255,0.2);
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(10px);
      color: white;
      cursor: pointer;
      transition: 0.3s;
    }

    .cta-btn:hover {
      background: rgba(0,150,255,0.3);
      transform: scale(1.05);
    }

  </style>
</head>

<body>

<section class="hero">

  <div class="bg-glow"></div>

  <!-- METRO IMAGE -->
  <img src="https://cdn-icons-png.flaticon.com/512/2972/2972185.png" class="metro" id="metro">

  <!-- CONTENT -->
  <div class="content">
    <h1 class="title" id="title">CityPulse</h1>

    <p class="subtitle" id="subtitle">
      An IoT-Enabled, Deep Learning and Machine Learning Framework for Predicting 
      Urban Traffic Disruption During Metro Construction
    </p>

    <button class="cta-btn">Explore Project</button>
  </div>

</section>

<script>
  const metro = document.getElementById("metro");
  const title = document.getElementById("title");
  const subtitle = document.getElementById("subtitle");

  window.addEventListener("scroll", () => {
    let scrollY = window.scrollY;

    // Metro moves horizontally then downward
    let xMove = scrollY * 0.5;
    let yMove = scrollY * 0.3;

    metro.style.transform = `translate(${xMove}px, ${yMove}px)`;

    // Trigger text animation
    if (scrollY > 100) {
      title.classList.add("show");
    }

    if (scrollY > 200) {
      subtitle.classList.add("show");
    }
  });
</script>

</body>
</html>