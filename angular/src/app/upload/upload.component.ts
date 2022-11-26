import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { FileuploadService } from '../fileupload.service';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent implements OnInit {
  //formData = {title:"",content:"",cover:""};
  fileName!: string ;
  loading: boolean = false;
  submit=false;
  
   
  constructor(private router: Router, private http: HttpClient,private fileupload : FileuploadService){ }
  fileChangeListener($res: any){
    const file:File = $res.target.files[0];
    this.fileName = file.name;
    const formData =new FormData();
    formData.append("files", file, this.fileName);
    return this.http.post<any>("http://localhost:5000/upload", formData).subscribe(res =>{
      console.log(res)
    })
  }

  
  onUpload(){
    this.loading = !this.loading;
    console.log("saved successfully")
    
  }

  ngOnInit(): void {

  }

  //public file : any;
  //public busy: boolean = false;
  //public postForm = new FormGroup({
    //title: new FormControl('', Validators.required),
    //content: new FormControl('',  Validators.required),
    //cover: new FormControl('',  Validators.required),  
  //});

  //public fileChangeListener($event: any){
    //getting the image or files
    //this.file = $event.target["files"];
    //console.log(this.file);
 // }

 // public addPost(res: any){
   // const file:File = res.target.files[0];
    //this.fileName = file.name;
    //this.busy = true;
    //const formData = new FormData();
    //formData.append("files",file,this.fileName);
    //this.fileupload.addPost(res, this.formData.title,this.formData.content, this.formData.cover,this.formData).subscribe(res => {
      //this.busy = false;
      //console.log(res);
      //this.router.navigate(["/prediction"]);
    //});


    
  //}
}