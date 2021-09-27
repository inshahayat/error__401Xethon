<?php

session_start();

// 2. Unset all the session variables
unset($_SESSION['MEMBER_ID']);
unset($_SESSION['Name']);
unset($_SESSION['Email']);

?>
<script type="text/javascript">
alert("Logout Successfully...");
window.location = "../index.html";
</script>

?>