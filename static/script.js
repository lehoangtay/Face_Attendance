function fetchAttendance() {
    fetch('/attendance')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('attendance_data');
            tbody.innerHTML = '';
            data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td><td>${row[3]}</td>`;
                tbody.appendChild(tr);
            });
        });
}
setInterval(fetchAttendance, 5000); // Cập nhật mỗi 2 giây
fetchAttendance();
