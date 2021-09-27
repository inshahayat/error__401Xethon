<?php
include("../authentication/authenticate.php");
include("../authentication/session.php");
if (isset($_POST['submit'])) {
 
  $email = trim($_POST['email']);
  $upass = trim($_POST['password']);
  $h_upass = md5($upass);

if ($upass == ''){
     ?>    <script type="text/javascript">
                alert("Password is missing!");
                window.location = "../login.php";
                </script>
        <?php
}else{

	    $sql = "SELECT * FROM user_signup WHERE email =  '" . $email . "' AND password =  '" . $h_upass . "'";
        $result = mysqli_query($con, $sql);
 
        if ($result){
             //get the number of results based n the sql statement
        $numrows = mysqli_num_rows($result);
    
        //check the number of result, if equal to one  
        //IF theres a result
            if ($numrows == 1) {
                //store the result to a array and passed to variable found_user
                $found_user  = mysqli_fetch_array($result);
 
                //fill the result to session variable
				$_SESSION['logged_in'] = TRUE;
                $_SESSION['MEMBER_ID']  = $found_user['signup_id'];
                $_SESSION['Name'] = $found_user['name'];
                $_SESSION['Email']  =  $found_user['email'];
          
             ?>    <script type="text/javascript">
                      //then it will be redirected to index.php
                      window.location = "../product.php";
                  </script>
             <?php        
          
        
            } else {
            //IF theres no result
			?>    <script type="text/javascript">
                alert("Email or password is incorrect.");
                window.location = "../login.php";
                </script>
        <?php
             
            }
 
         } else {
                 # code...
         die("Table Query failed: " );
        }
			
		}      
}      

?>
