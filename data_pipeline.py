import data_pipeline_1_join
import data_pipeline_2_clean
import data_pipeline_3_profile

def main():
    data_pipeline_1_join.main()
    data_pipeline_2_clean.main()
    data_pipeline_3_profile.main()
    print("Data pipeline completed!")

if __name__ == "__main__":
    main()
