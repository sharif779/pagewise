/**
 * pagewise chat widget. Embed with:
 *   <link rel="stylesheet" href=".../widget/chat.css">
 *   <script src=".../widget/chat.js" data-api="https://your-api.example.com"></script>
 */
(function () {
  var scriptTag = document.currentScript;
  var apiBase = (scriptTag && scriptTag.getAttribute("data-api")) || "";
  if (!apiBase) {
    console.error("pagewise widget: missing data-api attribute on the script tag.");
    return;
  }
  apiBase = apiBase.replace(/\/+$/, "");

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  var bubble = el("button", "pagewise-bubble", "?");
  var panel = el("div", "pagewise-panel");
  var header = el("div", "pagewise-header", "Ask about this document");
  var messages = el("div", "pagewise-messages");
  var inputRow = el("div", "pagewise-inputrow");
  var input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Ask a question...";
  var sendBtn = el("button", null, "Send");

  inputRow.appendChild(input);
  inputRow.appendChild(sendBtn);
  panel.appendChild(header);
  panel.appendChild(messages);
  panel.appendChild(inputRow);

  function addMessage(text, who, page) {
    var msg = el("div", "pagewise-msg pagewise-" + who, text);
    if (page) {
      msg.appendChild(el("span", "pagewise-page", "Page " + page));
    }
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
  }

  async function ask(question) {
    addMessage(question, "user");
    try {
      var res = await fetch(apiBase + "/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });
      var data = await res.json();
      if (!res.ok) {
        addMessage(data.detail || "Something went wrong.", "bot");
        return;
      }
      addMessage(data.answer, "bot", data.page);
    } catch (err) {
      addMessage("Network error. Please try again.", "bot");
    }
  }

  sendBtn.addEventListener("click", function () {
    var q = input.value.trim();
    if (!q) return;
    input.value = "";
    ask(q);
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") sendBtn.click();
  });
  bubble.addEventListener("click", function () {
    panel.classList.toggle("pagewise-open");
  });

  document.addEventListener("DOMContentLoaded", function () {
    document.body.appendChild(bubble);
    document.body.appendChild(panel);
  });
  if (document.readyState !== "loading") {
    document.body.appendChild(bubble);
    document.body.appendChild(panel);
  }
})();
