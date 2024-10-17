// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}" // Adjust this to match the file extensions and paths used in your project
  ],
  theme: {
    extend:{
      colors: {
        customGray:'#EBECED',
      },
    },
    extend: {},
  },
  plugins: [function ({ addUtilities }) {
    addUtilities({
      '.no-scrollbar': {
        '-ms-overflow-style': 'none',  /* Internet Explorer 10+ */
        'scrollbar-width': 'none',     /* Firefox */
      },
      '.no-scrollbar::-webkit-scrollbar': {
        display: 'none',               /* WebKit (Chrome/Safari) */
      },
    });
  },],
}
