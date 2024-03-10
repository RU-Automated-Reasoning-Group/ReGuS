import os
from PIL import Image
from robot import KarelRobot
import pdb

def store_imgs(im_dict, state_list, store_path, im_size):
    for state_id, state in enumerate(state_list):
        new_im = Image.new('RGB', (im_size, im_size))

        # print for each position
        for w in range(im_size):
            for h in range(im_size):
                cur_pos = state[w, h]
                if cur_pos[0]:
                    new_im.paste(im_dict['north'], (w*im_size, h*im_size))
                elif cur_pos[1]:
                    new_im.paste(im_dict['south'], (w*im_size, h*im_size))
                elif cur_pos[2]:
                    new_im.paste(im_dict['west'], (w*im_size, h*im_size))
                elif cur_pos[3]:
                    new_im.paste(im_dict['east'], (w*im_size, h*im_size))
                elif cur_pos[4]:
                    new_im.paste(im_dict['wall'], (w*im_size, h*im_size))
                elif cur_pos[5]:
                    new_im.paste(im_dict['empty'], (w*im_size, h*im_size))
                else:
                    new_im.paste(im_dict['marker'], (w*im_size, h*im_size))

        # store
        new_im.save(os.path.join(store_path, '{}.png'.format(state_id)))


if __name__ == '__main__':
    # get image library
    im_name_dict = {'north': 'agent_0.PNG', 
                    'east': 'agent_1.PNG', 
                    'south': 'agent_2.PNG', 
                    'west': 'agent_3.PNG', 
                    'empty': 'blank.PNG', 
                    'marker': 'agent_0.PNG', }
    im_dict = {Image.open(os.path.join('asset' ,im_name_dict[k])) for k in im_name_dict}

    # get state
    task = 'topOff'
    seeds = [0, 1000, 2000, 3000, 4000]
    test_seeds = [10000 + 1000 * i for i in range(10)]

    state_list = [KarelRobot(task, e).init_state for e in seeds+test_seeds]

    # store image
    pdb.set_trace()
    store_path = 'imgs'
    store_imgs(im_dict, state_list, store_path, 1)