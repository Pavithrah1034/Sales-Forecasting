import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';



@Injectable({
  providedIn: 'root'
})
export class FileuploadService {

  constructor(private router:Router,private http:HttpClient) { }
  public server:string = "http://localhost:5000"
  fileName: string | undefined;


  isAuthenticated():boolean{
    if(sessionStorage.getItem('token')!==null){
      return true;
    }
    return false;
  }
canAccess(){
if(!this.isAuthenticated()){
  this.router.navigate(['/login'])
}
}
canAutheticated(){
  if(this.isAuthenticated()){
    console.log("Successfully logged in!")
  }
  }

register(name:string,mail:string,pwd:string){
  return this.http.post<{idToken:string}>("https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyAT4xSVOmWIy8ElVP5OKSVYJIPdOLx36P4",
  {displayName:name,email:mail,password:pwd});
}
login(mail:string,pwd:string){
  return this.http.post<{idToken:string}>("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyAT4xSVOmWIy8ElVP5OKSVYJIPdOLx36P4",
  {email:mail,password:pwd});
}

storeToken(token:string){
  sessionStorage.setItem('token',token)
}

removeToken(){
  sessionStorage.removeItem('token');
  this.router.navigate(['/'])
}


}