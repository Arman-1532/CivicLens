import { useState } from 'react';

const Field = ({ label, name, form, errors, handleChange, isLoading, type = 'text', required = false, placeholder = '', hint = '' }) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-1">
      {label} {required && <span className="text-red-500">*</span>}
    </label>
    <input
      id={name}
      name={name}
      type={type}
      value={form[name]}
      onChange={handleChange}
      placeholder={placeholder}
      disabled={isLoading}
      className={`w-full px-3 py-2.5 border rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors
        ${errors[name] ? 'border-red-400 bg-red-50' : 'border-gray-300 bg-white'}`}
    />
    {errors[name] && <p className="mt-1 text-xs text-red-600">{errors[name]}</p>}
    {hint && !errors[name] && <p className="mt-1 text-xs text-gray-400">{hint}</p>}
  </div>
);

function ComplaintForm({ onSubmit, isLoading }) {
  const [form, setForm] = useState({
    citizen_name: '',
    citizen_email: '',
    citizen_phone: '',
    location: '',
    complaint_text: '',
  });
  const [errors, setErrors] = useState({});

  const charCount = form.complaint_text.length;
  const maxChars = 5000;

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === 'complaint_text' && value.length > maxChars) return;
    setForm((prev) => ({ ...prev, [name]: value }));
    if (errors[name]) setErrors((prev) => ({ ...prev, [name]: '' }));
  };

  const validate = () => {
    const newErrors = {};
    if (!form.citizen_name.trim()) newErrors.citizen_name = 'Full name is required';
    if (!form.citizen_email.trim()) newErrors.citizen_email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(form.citizen_email)) newErrors.citizen_email = 'Enter a valid email';
    if (!form.location.trim()) newErrors.location = 'Location is required';
    if (form.complaint_text.trim().length < 10) newErrors.complaint_text = 'Complaint must be at least 10 characters';
    return newErrors;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const validationErrors = validate();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
    const payload = {
      citizen_name: form.citizen_name.trim(),
      citizen_email: form.citizen_email.trim(),
      citizen_phone: form.citizen_phone.trim() || null,
      location: form.location.trim(),
      complaint_text: form.complaint_text.trim(),
    };
    onSubmit(payload);
  };

  const handleClear = () => {
    setForm({ citizen_name: '', citizen_email: '', citizen_phone: '', location: '', complaint_text: '' });
    setErrors({});
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Citizen Info Section */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="sm:col-span-2">
          <Field
            label="Full Name"
            name="citizen_name"
            required
            placeholder="e.g. Ali Hassan"
            form={form}
            errors={errors}
            handleChange={handleChange}
            isLoading={isLoading}
          />
        </div>
        <Field
          label="Email Address"
          name="citizen_email"
          type="email"
          required
          placeholder="ali@example.com"
          form={form}
          errors={errors}
          handleChange={handleChange}
          isLoading={isLoading}
        />
        <Field
          label="Phone Number"
          name="citizen_phone"
          placeholder="01700000000"
          hint="Optional"
          form={form}
          errors={errors}
          handleChange={handleChange}
          isLoading={isLoading}
        />
      </div>

      <Field
        label="Location / Area"
        name="location"
        required
        placeholder="e.g. Mirpur-10, Dhaka"
        form={form}
        errors={errors}
        handleChange={handleChange}
        isLoading={isLoading}
      />

      {/* Complaint Text */}
      <div>
        <label htmlFor="complaint_text" className="block text-sm font-medium text-gray-700 mb-1">
          Complaint Details <span className="text-red-500">*</span>
        </label>
        <textarea
          id="complaint_text"
          name="complaint_text"
          value={form.complaint_text}
          onChange={handleChange}
          placeholder="Describe your complaint in detail&#10;&#10;Example: The water supply in our area has been irregular for the past two weeks. Despite multiple complaints to local authorities, no action has been taken."
          rows={6}
          disabled={isLoading}
          className={`w-full px-3 py-2.5 border rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-colors
            ${errors.complaint_text ? 'border-red-400 bg-red-50' : 'border-gray-300 bg-white'}`}
        />
        <div className="flex justify-between items-center mt-1">
          {errors.complaint_text ? (
            <p className="text-xs text-red-600">{errors.complaint_text}</p>
          ) : (
            <p className="text-xs text-gray-400">Our AI will automatically classify your complaint</p>
          )}
          <span className={`text-xs ${charCount > maxChars * 0.9 ? 'text-orange-500' : 'text-gray-400'}`}>
            {charCount} / {maxChars}
          </span>
        </div>
      </div>

      {/* Buttons */}
      <div className="flex gap-3 pt-1">
        <button
          type="submit"
          disabled={isLoading}
          className={`flex-1 py-3 px-6 rounded-lg font-medium text-white transition-all duration-200
            ${!isLoading ? 'bg-blue-600 hover:bg-blue-700 active:scale-[0.98]' : 'bg-blue-400 cursor-not-allowed'}`}
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Submitting...
            </span>
          ) : 'Submit Complaint'}
        </button>
        <button
          type="button"
          onClick={handleClear}
          disabled={isLoading}
          className="py-3 px-6 rounded-lg font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors"
        >
          Clear
        </button>
      </div>
    </form>
  );
}

export default ComplaintForm;
