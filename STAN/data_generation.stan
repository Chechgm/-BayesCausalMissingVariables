data{
    int<lower=1> N;
}
parameters{
    real beta_0_y; 
    real beta_1_y;
    real beta_2_y;
    real beta_u_y;
    real beta_z_y;
    real<lower=0> sigma_y;
    
    real beta_0_u; 
    real beta_1_u;
    real beta_2_u; 
    real<lower=0> sigma_u;
}
model { 
    beta_0_y ~ std_normal();
    beta_1_y ~ std_normal();
    beta_2_y ~ std_normal();
    beta_u_y ~ std_normal();
    beta_z_y ~ std_normal();
    sigma_y ~ std_normal();
    
    beta_0_u ~ std_normal();
    beta_1_u ~ std_normal();
    beta_2_u ~ std_normal();
    sigma_u ~ std_normal();
} 
generated quantities { 
    vector[N] y;
    vector[N] x_1;
    vector[N] x_2;
    vector[N] z;
    vector[N] u;
    for(n in 1:N) {
        x_1[n] = normal_rng(3, 2);
        x_2[n] = normal_rng(-3, 2);
        z[n] = bernoulli_rng(0.4);
        u[n] = normal_rng(beta_0_u + beta_1_u*x_1[n] + beta_2_u*x_2[n], sigma_u);
        y[n] = normal_rng(beta_0_y + beta_1_y*x_1[n] + beta_2_y*x_2[n] + beta_u_y*u[n] + beta_z_y*z[n], sigma_y); 
    }

} 