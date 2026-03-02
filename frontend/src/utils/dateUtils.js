/**
 * Formats an ISO date string to a human-readable Bangladeshi time string (UTC+6).
 * @param {string} dateString - ISO date string from backend.
 * @returns {string} Formatted date and time.
 */
export const formatToBDDate = (dateString) => {
    if (!dateString) return '';

    try {
        // Ensure the date string is treated as UTC if it doesn't have a timezone indicator
        let normalizedDateString = dateString;
        if (typeof dateString === 'string' && !dateString.includes('Z') && !dateString.includes('+') && !dateString.includes('-')) {
            // Check if it's a typical ISO string without TZ
            if (dateString.includes('T')) {
                normalizedDateString = `${dateString}Z`;
            }
        }

        const date = new Date(normalizedDateString);

        // Check if date is valid
        if (isNaN(date.getTime())) return dateString;

        return new Intl.DateTimeFormat('en-GB', {
            timeZone: 'Asia/Dhaka',
            day: '2-digit',
            month: 'short',
            year: 'numeric'
        }).format(date);
    } catch (error) {
        console.error('Error formatting date:', error);
        return dateString;
    }
};
