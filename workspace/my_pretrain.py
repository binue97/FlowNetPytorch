import argparse
import os
import yaml

def main():
  parser = argparse.ArgumentParser(description="Read a YAML config file.")
  parser.add_argument("-i", "--config", type=str, help="Path to the YAML configuration file.")
  args = parser.parse_args()

  # Read YAML file
  config = None
  try:
    with open(args.config, "r") as f:
      config = yaml.safe_load(f)
      print(config)
  except yaml.YAMLError as e:
    print(f"Error parsing YAML file: {e}")



if __name__ == "__main__":
  main()
