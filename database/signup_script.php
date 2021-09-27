<?php
include("../authentication/authenticate.php");

$name = "";
$email = "";
$password = "";
$contact = "";
$city = "";
$address = "";

$name = mysqli_real_escape_string($con, $_POST['name']);
$email = mysqli_real_escape_string($con, $_POST['email']);
$password = mysqli_real_escape_string($con, $_POST['password']);
$contact = mysqli_real_escape_string($con, $_POST['contact']);
$city = mysqli_real_escape_string($con, $_POST['city']);
$address = mysqli_real_escape_string($con, $_POST['address']);

$sql=mysqli_query($con, "SELECT * FROM user_signup where email='$email'");

if(isset($_POST['submit']))
{

    if(mysqli_num_rows($sql)>0)
    {
        echo "Email already exists.";
    }
    else
    {
        $pass = md5($password); //use for encrypt the password
        $query = "INSERT INTO user_signup(name, email, password, mobile_no, city, address) VALUES ('$name', '$email', '$pass', '$contact', '$city', '$address')";
        $sql=mysqli_query($con, $query) or error_reporting(E_ALL);
            header ("Location: ../login.php?status=success");
    }
}
else
{
    echo "Please try again later.";
}
?>