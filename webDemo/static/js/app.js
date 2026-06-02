document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.view-section');
    
    // Registration elements
    const regSetup = document.getElementById('reg-setup');
    const regActive = document.getElementById('reg-active');
    const regSuccess = document.getElementById('reg-success');
    const regNameInput = document.getElementById('reg-name');
    const startRegNormalBtn = document.getElementById('start-reg-normal-btn');
    const startRegMaskedBtn = document.getElementById('start-reg-masked-btn');
    const stopRegBtn = document.getElementById('stop-reg-btn');
    const regVideo = document.getElementById('reg-video');
    const regPlaceholder = document.getElementById('reg-placeholder');
    const regInstruction = document.getElementById('reg-instruction');
    const regProgressBar = document.getElementById('reg-progress-bar');
    const regProgressText = document.getElementById('reg-progress-text');
    
    // Attendance elements
    const startAttBtn = document.getElementById('start-att-btn');
    const stopAttBtn = document.getElementById('stop-att-btn');
    const attVideo = document.getElementById('att-video');
    const attPlaceholder = document.getElementById('att-placeholder');
    const attHistory = document.getElementById('att-history');
    
    let statePollingInterval = null;
    let currentRegType = 'normal';

    // Navigation Logic
    navBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            // Stop current operations when navigating
            await fetch('/api/stop', { method: 'POST' });
            
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const targetId = btn.getAttribute('data-target');
            sections.forEach(sec => {
                sec.classList.remove('active');
                if(sec.id === targetId) sec.classList.add('active');
            });
            
            // Reset UI states
            resetRegistrationUI();
            resetAttendanceUI();
        });
    });

    // --- API Calls ---
    async function startRegistration(regType = 'normal') {
        currentRegType = regType;
        const name = regNameInput.value.trim();
        if (!name) {
            alert('Vui lòng nhập tên!');
            return;
        }
        
        try {
            const res = await fetch('/api/start_register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, type: regType })
            });
            const data = await res.json();
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Update UI
            regSetup.classList.add('hidden');
            regActive.classList.remove('hidden');
            regPlaceholder.classList.add('hidden');
            regVideo.src = '/video_feed?' + new Date().getTime(); // cache bust
            regVideo.classList.remove('hidden');
            
            startStatePolling();
            
        } catch (e) {
            console.error('Lỗi khi bắt đầu đăng ký:', e);
        }
    }

    async function stopRegistration() {
        await fetch('/api/stop', { method: 'POST' });
        resetRegistrationUI();
    }

    async function startAttendance() {
        try {
            await fetch('/api/start_attendance', { method: 'POST' });
            
            startAttBtn.classList.add('hidden');
            stopAttBtn.classList.remove('hidden');
            
            attPlaceholder.classList.add('hidden');
            attVideo.src = '/video_feed?' + new Date().getTime();
            attVideo.classList.remove('hidden');
            
            startStatePolling();
            
        } catch (e) {
            console.error('Lỗi khi bắt đầu điểm danh:', e);
        }
    }

    async function stopAttendance() {
        await fetch('/api/stop', { method: 'POST' });
        resetAttendanceUI();
    }

    // --- State Polling ---
    function startStatePolling() {
        if (statePollingInterval) clearInterval(statePollingInterval);
        statePollingInterval = setInterval(pollState, 500);
    }

    function stopStatePolling() {
        if (statePollingInterval) clearInterval(statePollingInterval);
        statePollingInterval = null;
    }

    async function pollState() {
        try {
            const res = await fetch('/api/state');
            const data = await res.json();
            
            if (data.mode === 'REGISTER') {
                if (data.reg_done) {
                    // Registration complete
                    stopStatePolling();
                    regActive.classList.add('hidden');
                    regVideo.classList.add('hidden');
                    regVideo.src = "";
                    regPlaceholder.classList.remove('hidden');
                    
                    const successTitle = document.getElementById('reg-success-title');
                    const successMsg = document.getElementById('reg-success-msg');
                    const actions = document.getElementById('reg-success-actions');
                    const finalActions = document.getElementById('reg-success-final-actions');
                    
                    if (currentRegType === 'normal') {
                        successTitle.textContent = "Đăng ký mặt thường thành công!";
                        successMsg.textContent = "Dữ liệu khuôn mặt đã được lưu. Bạn có muốn bổ sung ảnh đeo khẩu trang để hệ thống nhận diện tốt hơn không?";
                        actions.classList.remove('hidden');
                        finalActions.classList.add('hidden');
                    } else {
                        successTitle.textContent = "Đăng ký thành công!";
                        successMsg.textContent = "Toàn bộ dữ liệu khuôn mặt đã được lưu và hệ thống đang huấn luyện lại. Bạn có thể sử dụng chức năng Điểm danh ngay bây giờ.";
                        actions.classList.add('hidden');
                        finalActions.classList.remove('hidden');
                    }
                    
                    regSuccess.classList.remove('hidden');
                } else {
                    regInstruction.textContent = data.instruction;
                    const percent = data.target > 0 ? (data.captured / data.target) * 100 : 0;
                    regProgressBar.style.width = `${percent}%`;
                    regProgressText.textContent = `${data.captured} / ${data.target} ảnh`;
                }
            } else if (data.mode === 'ATTENDANCE') {
                updateAttendanceHistory(data.history);
            }
            
        } catch (e) {
            console.error('Lỗi khi poll state:', e);
        }
    }

    // --- UI Helpers ---
    function resetRegistrationUI() {
        stopStatePolling();
        regSetup.classList.remove('hidden');
        regActive.classList.add('hidden');
        regSuccess.classList.add('hidden');
        regVideo.classList.add('hidden');
        regVideo.src = "";
        regPlaceholder.classList.remove('hidden');
        regNameInput.value = '';
        regProgressBar.style.width = '0%';
        regProgressText.textContent = '0 / 0 ảnh';
    }

    function resetAttendanceUI() {
        stopStatePolling();
        startAttBtn.classList.remove('hidden');
        stopAttBtn.classList.add('hidden');
        attVideo.classList.add('hidden');
        attVideo.src = "";
        attPlaceholder.classList.remove('hidden');
    }

    function updateAttendanceHistory(historyList) {
        if (!historyList || historyList.length === 0) {
            attHistory.innerHTML = '<li class="empty-state">Chưa có ai điểm danh...</li>';
            return;
        }
        
        attHistory.innerHTML = historyList.map(item => `
            <li>
                <span class="history-name">${item.name}</span>
                <span class="history-time">${item.time}</span>
            </li>
        `).join('');
    }

    // --- Event Listeners ---
    startRegNormalBtn.addEventListener('click', () => startRegistration('normal'));
    startRegMaskedBtn.addEventListener('click', () => startRegistration('masked'));
    stopRegBtn.addEventListener('click', stopRegistration);
    regNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') startRegistration('normal');
    });
    
    document.getElementById('continue-mask-btn').addEventListener('click', () => {
        regSuccess.classList.add('hidden');
        startRegistration('masked');
    });
    
    startAttBtn.addEventListener('click', startAttendance);
    stopAttBtn.addEventListener('click', stopAttendance);
});
