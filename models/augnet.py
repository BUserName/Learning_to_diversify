import torch
import torch.nn as nn
import torch.nn.functional as F


class AugNet(nn.Module):
    def __init__(self, noise_lv):
        super(AugNet, self).__init__()
        ############# Trainable Parameters
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(3,216,216))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, 216, 216))
        nn.init.normal_(self.shift_mean, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(3, 212, 212))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, 212, 212))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(3, 208, 208))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, 208, 208))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.shift_var4 = nn.Parameter(torch.empty(3, 220, 220))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(3, 220, 220))
        nn.init.normal_(self.shift_mean4, 0, 0.1)

        # self.shift_var5 = nn.Parameter(torch.empty(3, 206, 206))
        # nn.init.normal_(self.shift_var5, 1, 0.1)
        # self.shift_mean5 = nn.Parameter(torch.zeros(3, 206, 206))
        # nn.init.normal_(self.shift_mean5, 0, 0.1)
        #
        # self.shift_var6 = nn.Parameter(torch.empty(3, 204, 204))
        # nn.init.normal_(self.shift_var6, 1, 0.5)
        # self.shift_mean6 = nn.Parameter(torch.zeros(3, 204, 204))
        # nn.init.normal_(self.shift_mean6, 0, 0.1)

        # self.shift_var7 = nn.Parameter(torch.empty(3, 202, 202))
        # nn.init.normal_(self.shift_var7, 1, 0.5)
        # self.shift_mean7 = nn.Parameter(torch.zeros(3, 202, 202))
        # nn.init.normal_(self.shift_mean7, 0, 0.1)

        self.norm = nn.InstanceNorm2d(3)

        ############## Fixed Parameters (For MI estimation
        self.spatial = nn.Conv2d(3, 3, 9).cuda()
        self.spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

        self.spatial2 = nn.Conv2d(3, 3, 13).cuda()
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

        self.spatial3 = nn.Conv2d(3, 3, 17).cuda()
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()


        self.spatial4 = nn.Conv2d(3, 3, 5).cuda()
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()


        # self.spatial5 = nn.Conv2d(3, 3, 19).cuda()
        # self.spatial_up5 = nn.ConvTranspose2d(3, 3, 19).cuda()
        # +
        # list(self.spatial5.parameters()) + list(self.spatial_up5.parameters())
        # #+
        #
        # self.spatial6 = nn.Conv2d(3, 3, 21).cuda()
        # self.spatial_up6 = nn.ConvTranspose2d(3, 3, 21).cuda()
        # list(self.spatial6.parameters()) + list(self.spatial_up6.parameters())
        # self.spatial7 = nn.Conv2d(3, 3, 23).cuda()
        # self.spatial_up7= nn.ConvTranspose2d(3, 3, 23).cuda()
        # list(self.spatial7.parameters()) + list(self.spatial_up7.parameters())
        self.color = nn.Conv2d(3, 3, 1).cuda()

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters()) +
                          list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
                          list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
                          list(self.spatial4.parameters()) + list(self.spatial_up4.parameters())
                          ):
            param.requires_grad=False

    def forward(self, x, estimation=False):
        if not estimation:
            spatial = nn.Conv2d(3, 3, 9).cuda()
            spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

            spatial2 = nn.Conv2d(3, 3, 13).cuda()
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

            spatial3 = nn.Conv2d(3, 3, 17).cuda()
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()

            spatial4 = nn.Conv2d(3, 3, 5).cuda()
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()

            # spatial5 = nn.Conv2d(3, 3, 19).cuda()
            # spatial_up5 = nn.ConvTranspose2d(3, 3, 19).cuda()
            #
            # spatial6 = nn.Conv2d(3, 3, 21).cuda()
            # spatial_up6 = nn.ConvTranspose2d(3, 3, 21).cuda()

            # spatial7 = nn.Conv2d(3, 3, 23).cuda()
            # spatial_up7 = nn.ConvTranspose2d(3, 3, 23).cuda()

            color = nn.Conv2d(3,3,1).cuda()
            weight = torch.randn(5)

            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(F.dropout(color(x), p=.2))

            x_sdown = spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(spatial_up(x_sdown))
            #
            x_s2down = spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(spatial_up2(x_s2down))
            #
            #
            x_s3down = spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(spatial_up3(x_s3down))

            #
            x_s4down = spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(spatial_up4(x_s4down))

            # x_s5down = spatial5(x)
            # x_s5down = self.shift_var5 * self.norm(x_s5down) + self.shift_mean5
            # x_s5 = torch.tanh(spatial_up5(x_s5down))+ weight[5] * x_s5

            # x_s6down = spatial6(x)
            # x_s6down = self.shift_var6 * self.norm(x_s6down) + self.shift_mean6
            # x_s6 = torch.tanh(spatial_up6(x_s6down))+ weight[6] * x_s6

            # x_s7down = spatial7(x)
            # x_s7down = self.shift_var7 * self.norm(x_s7down) + self.shift_mean7
            # x_s7 = torch.tanh(spatial_up7(x_s7down))+ weight[7] * x_s7

            output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2+ weight[3] * x_s3 + weight[4]*x_s4) / weight.sum()
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            #
            x_sdown = self.spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial_up(x_sdown))
            #
            x_s2down = self.spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial_up2(x_s2down))

            x_s3down = self.spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial_up3(x_s3down))

            x_s4down = self.spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(self.spatial_up4(x_s4down))

            # x_s5down = self.spatial5(x)
            # x_s5down = self.shift_var5 * self.norm(x_s5down) + self.shift_mean5
            # x_s5 = torch.tanh(self.spatial_up5(x_s5down)) + x_s5

            # x_s6down = self.spatial6(x)
            # x_s6down = self.shift_var6 * self.norm(x_s6down) + self.shift_mean6
            # x_s6 = torch.tanh(self.spatial_up6(x_s6down))+ x_s6

            # x_s7down = self.spatial7(x)
            # x_s7down = self.shift_var7 * self.norm(x_s7down) + self.shift_mean7
            # x_s7 = torch.tanh(self.spatial_up7(x_s7down))+ x_s7

            output = (x_c + x_s + x_s2 + x_s3 + x_s4) / 5
        return output
