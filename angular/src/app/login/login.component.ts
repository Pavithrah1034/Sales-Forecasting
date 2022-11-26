import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

import { FileuploadService } from '../fileupload.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {


  formdata = {email:"",password:""};
  submit=false;
  loading=false;
  errorMessage="";
  
  constructor(private fileupload:FileuploadService, private router: Router) { }

  ngOnInit(): void {
    this.fileupload.canAutheticated();
    
  }

  onSubmit(){
    this.loading=true;
    //call login service
    this.fileupload.login(this.formdata.email,this.formdata.password)
    
    .subscribe({
        next:data=>{
            //store token
            this.fileupload.storeToken(data.idToken);
            console.log('logged user token is '+data.idToken);
            this.fileupload.canAutheticated();
        },
        error:data=>{
            if (data.error.error.message=="INVALID_PASSWORD" || data.error.error.message=="INVALID_EMAIL") {
                this.errorMessage = "Invalid Credentials!";
            } else{
                this.errorMessage = "Unknown error when logging into this account!";
            }
        }
    }).add(()=>{
        this.loading =false;
        console.log('login process completed!');

    })
    this.router.navigate(['/upload'])
  }
  
}