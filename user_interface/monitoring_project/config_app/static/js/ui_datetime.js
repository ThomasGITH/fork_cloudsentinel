(function () {
  "use strict";
  const formatter = new Intl.DateTimeFormat("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", hourCycle: "h23",
    timeZone: "UTC", timeZoneName: "short"
  });
  window.CloudSentinel = window.CloudSentinel || {};
  window.CloudSentinel.formatDateTime = function (value, fallback) {
    if (value === null || value === undefined || value === "") return fallback || "—";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : formatter.format(parsed);
  };
})();
