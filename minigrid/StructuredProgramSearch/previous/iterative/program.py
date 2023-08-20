
class SimpleProgram(nn.Module):
    def __init__(self, lines, act_dim):
        super().__init__()

        self.lines = lines
        self.M = Variable(torch.rand(lines, act_dim))
        
    # NOTE: obs should be the raw karel state
    # NOTE: for simple Program, obs is not needed
    def forward(self, obs):

        # NOTE: for other conditions / actions in complex programs
        #       we should use obs to determine which branch to execute

        # return a sequence of actions
        actions = self.M.softmax(dim=1).argmax(dim=1)

        return actions
