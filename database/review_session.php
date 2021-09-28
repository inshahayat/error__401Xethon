<?php
 session_start();
 if(isset($_POST['write']))
{   
    if (!isset($_SESSION['loggedin']))
        {
                 header('Location: ../login.php');
            }
        else
            {
                 if (isset($_POST['write']))
                     {
                         header('location: ../review.php');       
                         session_destroy();
                     }
            }   
}
?>