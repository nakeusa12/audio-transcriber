"""Penyimpanan job lintas sesi. Module ini di-import sekali oleh Python
(sys.modules) sehingga globalnya BERTAHAN across rerun/reconnect/tutup tab —
beda dengan st.session_state yang mati bersama sesi browser."""

JOB = None
