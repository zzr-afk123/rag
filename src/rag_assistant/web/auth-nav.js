/* 身份导航：所有页面共用。
 * - 未登录 -> 跳登录页（带上 next，登录后回到原页面）
 * - 按角色显示/隐藏带 data-role="dept" 的入口（学生、访客看不到部门看板/运维台）
 * - 往 #userBox 里放「姓名 · 角色」和退出按钮
 * 注意：隐藏只是体验层面，真正的拦截在服务端中间件（学生直接访问 /dept 会被 403）。
 */
(function () {
  'use strict';

  var STYLE_BTN = 'border:1px solid currentColor;background:transparent;color:inherit;' +
    'border-radius:999px;padding:3px 10px;font-size:11.5px;cursor:pointer;margin-left:2px;';
  var STYLE_NAME = 'font-size:11.5px;opacity:.85;white-space:nowrap;';

  function redirectToLogin() {
    if (location.pathname === '/login') return;
    var next = location.pathname + location.search;
    location.replace('/login?next=' + encodeURIComponent(next));
  }

  function renderUser(me) {
    document.querySelectorAll('[data-role="dept"]').forEach(function (el) {
      el.hidden = !me.can_view_dept;
    });
    var box = document.getElementById('userBox');
    if (!box) return;
    box.hidden = false;
    box.innerHTML = '';
    box.style.display = 'inline-flex';
    box.style.alignItems = 'center';
    box.style.gap = '8px';

    var name = document.createElement('span');
    name.style.cssText = STYLE_NAME;
    name.textContent = me.name + ' · ' + me.role_label;
    box.appendChild(name);

    var out = document.createElement('button');
    out.type = 'button';
    out.textContent = '退出';
    out.style.cssText = STYLE_BTN;
    out.addEventListener('click', function () {
      fetch('/auth/logout', { method: 'POST' })
        .catch(function () {})
        .then(function () { location.replace('/login'); });
    });
    box.appendChild(out);
  }

  fetch('/auth/me', { headers: { 'Accept': 'application/json' } })
    .then(function (resp) {
      if (resp.status === 401) { redirectToLogin(); return null; }
      if (!resp.ok) return null;
      return resp.json();
    })
    .then(function (payload) {
      if (payload && payload.user) renderUser(payload.user);
    })
    .catch(function () { /* 网络问题时交给页面自身处理 */ });
})();
