// Shared across signup/login/profile pages.
// Access token lives in localStorage; the refresh token is an httpOnly
// cookie the browser sends automatically — this JS never touches it.

const ACCESS_TOKEN_KEY = "gym_access_token";

function getAccessToken() {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

function setAccessToken(token) {
  localStorage.setItem(ACCESS_TOKEN_KEY, token);
}

function clearAccessToken() {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
}

// Fetch wrapper that attaches the access token and, on a 401, tries once
// to refresh it (via the httpOnly cookie) before giving up and sending
// the user to /login.
async function apiFetch(url, options = {}) {
  const doFetch = () => {
    const token = getAccessToken();
    const headers = { ...(options.headers || {}) };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    return fetch(url, { ...options, headers, credentials: "include" });
  };

  let response = await doFetch();

  if (response.status === 401) {
    const refreshed = await fetch("/api/refresh", {
      method: "POST",
      credentials: "include",
    });
    if (refreshed.ok) {
      const { access_token } = await refreshed.json();
      setAccessToken(access_token);
      response = await doFetch();
    } else {
      clearAccessToken();
      window.location.href = "/login";
      throw new Error("Session expired");
    }
  }

  return response;
}

async function logout() {
  await fetch("/api/logout", { method: "POST", credentials: "include" });
  clearAccessToken();
  window.location.href = "/login";
}

function showMessage(el, text, type) {
  el.textContent = text;
  el.className = `message show ${type}`;
}
