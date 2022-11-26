import {  NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { RegisterComponent } from './register/register.component';
import { LoginComponent } from './login/login.component';
import { UploadComponent } from './upload/upload.component';
import { PredictionComponent } from './prediction/prediction.component';


const routes: Routes = [
  { path: '', redirectTo:'login',pathMatch:'full'},
  { path: 'register', component: RegisterComponent},
  { path: 'upload', component: UploadComponent},
  { path: 'prediction', component: PredictionComponent},
  { path: 'login', component: LoginComponent}
];

@NgModule({
  imports: [
    RouterModule.forRoot(routes),
  ],
  exports: [RouterModule]
})
export class AppRoutingModule { }
