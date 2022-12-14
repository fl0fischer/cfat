from cfat.main import compute_gears_and_ctrlranges

if __name__ == "__main__":
    for participant_ID in range(1, 7):

        print(f"\n\n+++++++++++++++++++++++++++++++++++++\nPARTICIPANT U{participant_ID}\n++++++++++++++++++++++++++++++++++++++\n")
        DIRNAME_CFAT = f"_results/U{participant_ID}_0.002s"

        res_dict = compute_gears_and_ctrlranges(DIRNAME_CFAT, use_3xSTD_outliers=True)
        print(f"\nRESULT:\n{res_dict}")