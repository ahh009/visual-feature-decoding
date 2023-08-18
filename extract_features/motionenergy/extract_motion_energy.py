import utils
import argparse 

def main():
    parser = argparse.ArgumentParser(description="Extract motion energy")
    parser.add_argument('json_filepath')
    args = parser.parse_args() #this will grab from command line and put into arg parser object
    
    utils.movie_to_gray_array(args.json_filepath)
    utils.push_thru_pyramid(args.json_filepath)
    
if __name__ == "__main__":
    main()
        