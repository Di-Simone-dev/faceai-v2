# FaceAI v2

A modern React-based facial recognition and AI processing application built with Vite and TypeScript.

## Overview

FaceAI v2 is a web application that leverages AI and computer vision capabilities for facial analysis and processing. Built with modern web technologies, it provides a fast, type-safe development experience with hot module replacement (HMR).

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

## Preview Production Build

Preview the production build locally:

```bash
npm run preview
```

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

## Features

- ⚡ Lightning-fast development with Vite HMR
- 🎯 Type-safe development with TypeScript
- ⚛️ Optimized React performance with React Compiler
- 🔍 Code quality enforcement with ESLint
- 📦 Optimized production builds

## React Compiler

This project has the React Compiler enabled, which provides automatic optimization of React components. This feature:

- Automatically memoizes components and hooks
- Reduces unnecessary re-renders
- Improves overall application performance

**Note:** The React Compiler may impact Vite dev and build performance slightly.

For more information, see the [React Compiler documentation](https://react.dev/learn/react-compiler).

## Available Vite Plugins

The project supports two official Vite plugins for React:

- **[@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react)** - Uses Babel for Fast Refresh
- **[@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc)** - Uses SWC for Fast Refresh (faster than Babel)

## ESLint Configuration

For production applications, it's recommended to enable type-aware lint rules. To expand the ESLint configuration:

1. Update `parserOptions` in `eslint.config.ts`:
```typescript
parserOptions: {
  ecmaVersion: 'latest',
  sourceType: 'module',
  project: ['./tsconfig.json', './tsconfig.node.json'],
  tsconfigRootDir: __dirname,
}
```

2. Replace `tseslint.configs.recommended` with `tseslint.configs.recommendedTypeChecked` or `tseslint.configs.strictTypeChecked`

3. Optionally add `...tseslint.configs.stylisticTypeChecked`

4. Install and configure the React ESLint plugin:
```bash
npm install eslint-plugin-react --save-dev
```

Then update your config:
```typescript
import react from 'eslint-plugin-react'

export default tseslint.config({
  settings: { react: { version: '18.3' } },
  plugins: { react },
  rules: {
    ...react.configs.recommended.rules,
    ...react.configs['jsx-runtime'].rules,
  },
})
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint (if configured) |

## Browser Support

This project targets modern browsers with ES6+ support. For specific browser requirements, check the build configuration in `vite.config.ts`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Vite](https://vitejs.dev/)
- Powered by [React](https://react.dev/)
- Type-safe with [TypeScript](https://www.typescriptlang.org/)

## Contact

Project Link: [https://github.com/Di-Simone-dev/faceai-v2](https://github.com/Di-Simone-dev/faceai-v2)

---

**Note:** This is version 2 of the FaceAI project, rebuilt with modern tooling and improved architecture.
