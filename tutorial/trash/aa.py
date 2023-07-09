
\begin{align*}  
&\mathbb{V}_{t}[\hat{J}_{\mathrm{DR}}^{H+1-t}(\pi; \mathcal{D})]\\
&=\mathbb{E}_{t}\left[\left(\hat{J}_{\mathrm{DR}}^{H+1-t}\right)^2\right]-\Bigl(\mathbb{E}_{t}[V(s_t)]\Bigr)^2 \\
&=\mathbb{E}_{t}\left[\left(\hat{V}(s_t)+w_t\left(r_t+\gamma J_{\mathrm{DR}}^{H-t} - \hat{Q}(s_t, a_t)\right)\right)^2\right]-\mathbb{E}_{t}[V(s_t)^2]+\mathbb{V}_t[V(s_t)]\\
&=\mathbb{E}_{t}\left[\left(w_tQ(s_t, a_t)-w_t\hat{Q}(s_t, a_t)+\hat{V}(s_t)+w_t\left(r_t+\gamma J_{\mathrm{DR}}^{H-t}-Q(s_t, a_t)\right)\right)^2-V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
&=\mathbb{E}_{t}\left[\left(w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t)+w_t\left(r_t-R(s_t, a_t)\right)+w_t\gamma \left(J_{\mathrm{DR}}^{H-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
&=\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_t}\left[
\left(w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t)+w_t\left(r_t-R(s_t, a_t)\right)+w_t\gamma \left(J_{\mathrm{DR}}^{H-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right] \biggm\vert s_t, a_t\right]+\mathbb{V}_{t}[V(s_t)]\\
&=\mathbb{E}_{s_t}\left[\mathbb{E}_{a_t, r_t}\left[
\left(-w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t)\right)^2 - V(s_t)^2 \mid s_t\right]\right]+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_{t}^2\left(r_t -R(s_t, a_t)\right)^2\right]\right]\\
&+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_t^2\gamma^2\left(J_{\mathrm{DR}}^{H-t}(s_t, a_t)-\mathbb{E}_{t+1}[V(s_{t+1})]\right)^2\right]\right]+\mathbb{V}_{t}[V(s_t)]\\
&=\mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ -w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{DR}}^{H-t}]\right]+ \mathbb{V}_t[V(s_t)]\\
&=\mathbb{E}_{s_t}\left[\mathbb{V}_{a_t, r_t}\left[w_t(\hat{Q}(s_t, a_t)-Q(s_t, a_t)) \mid s_t\right]\right]+\mathbb{E}_{s_t, a_t}\left[{w_t}^2\mathbb{V}_{r_{t+1}}[r_t]\right] + \mathbb{E}_{s_t, a_t}\left[\gamma^2{w_t}^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{DR}}^{H-t}]\right] + \mathbb{V}_t[V(s_t)] 
\end{align*} 
