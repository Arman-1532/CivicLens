/**
 * Formats an ISO date string to a human-readable Bangladeshi time string (UTC+6).
 * @param {string} dateString - ISO date string from backend.
 * @returns {string} Formatted date and time.
 */
export const formatToBDTime = (dateString) => {
    if (!dateString) return '';

    try {
        const date = new Date(dateString);

        // Check if date is valid
        if (isNaN(date.getTime())) return dateString;

        return new Intl.DateTimeFormat('en-GB', {
            timeZone: 'Asia/Dhaka',
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }).format(date);
    } catch (error) {
        console.error('Error formatting date:', error);
        return dateString;
    }
};
