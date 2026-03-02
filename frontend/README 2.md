# CivicLens Frontend

React frontend for the AI-based Complaint Classification System.

## Tech Stack

- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router v6
- **HTTP Client**: Axios
- **State Management**: React Context API

## Project Structure

```
frontend/
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── ComplaintForm.jsx    # Complaint submission form
│   │   ├── ResultCard.jsx       # Classification result display
│   │   └── Navbar.jsx           # Navigation bar
│   │
│   ├── pages/
│   │   ├── Home.jsx             # Main page with form & results
│   │   ├── Dashboard.jsx        # Complaint history & stats
│   │   └── AdminPanel.jsx       # System health & config
│   │
│   ├── services/
│   │   └── api.js               # Axios API calls
│   │
│   ├── hooks/
│   │   └── useComplaint.js      # Custom hook for complaints
│   │
│   ├── context/
│   │   └── ComplaintContext.jsx # Global state management
│   │
│   ├── App.jsx                  # Main app component
│   ├── main.jsx                 # Entry point
│   ├── routes.jsx               # Route definitions
│   └── index.css                # Global styles + Tailwind
│
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Setup & Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will be available at: **http://localhost:5173**

### 3. Build for Production

```bash
npm run build
```

## Pages

### Home (`/`)
- Complaint submission form
- Real-time classification results
- Category information

### Dashboard (`/dashboard`)
- Complaint history table
- Statistics cards (total, urgency distribution)
- Category distribution chart

### Admin Panel (`/admin`)
- System health status
- Model information
- API endpoints reference

## Features

- ✅ **Complaint Form** with character count and validation
- ✅ **Real-time Classification** with loading states
- ✅ **Result Display** showing category, department, urgency, confidence
- ✅ **Dashboard** with statistics and history
- ✅ **Admin Panel** with system health monitoring
- ✅ **Responsive Design** with Tailwind CSS
- ✅ **Error Handling** with user-friendly messages

## API Integration

The frontend connects to the backend API at `http://localhost:8000`. Configure in `vite.config.js`:

```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```

## Environment Variables

Create a `.env` file (optional):

```env
VITE_API_URL=http://localhost:8000
```

## Screenshots

### Home Page
- Clean complaint submission form
- Instant AI classification results
- Category icons and descriptions

### Dashboard
- Statistics overview
- Complaint history table
- Category distribution

### Admin Panel
- System health status
- Model information
- API endpoints

## Development

```bash
# Start dev server
npm run dev

# Lint code
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## License

MIT License

