# FaceAI v2

A modern React-based facial recognition and AI processing application built with Vite and TypeScript.

## Tech Stack

- **React** - UI framework with React Compiler enabled for optimized performance
- **TypeScript** - Type-safe JavaScript development
- **Vite** - Next-generation frontend build tool with lightning-fast HMR
- **ESLint** - Code quality and consistency enforcement

## Prerequisites

- Node.js (v16 or higher recommended)
- npm or yarn package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Di-Simone-dev/faceai-v2.git
cd faceai-v2
```

2. Install dependencies:
```bash
npm install
```

## Development

Start the development server with hot module replacement:

```bash
npm run dev
```

The application will be available at `http://localhost:5173` (default Vite port).

## Build

Create a production-optimized build:

```bash
npm run build
```

The build output will be in the `dist` directory.



## Project Structure

```
faceai-v2/
├── public/          # Static assets
├── src/             # Source code
│   ├── components/  # React components
│   ├── hooks/       # Custom React hooks
│   ├── utils/       # Utility functions
│   └── App.tsx      # Main application component
├── index.html       # HTML entry point
├── package.json     # Dependencies and scripts
├── tsconfig.json    # TypeScript configuration
├── vite.config.ts   # Vite configuration
└── eslint.config.ts # ESLint configuration
```


## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |

## Browser Support

This project targets modern browsers with ES6+ support. For specific browser requirements, check the build configuration in `vite.config.ts`.
