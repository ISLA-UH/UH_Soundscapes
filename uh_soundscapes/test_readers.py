"""
Test the dataset readers
"""
import matplotlib.pyplot as plt

import uh_soundscapes.astra_reader as asr, esc50_reader as ecr, orex_reader as oxr, shared_explosions_reader as ser


def test_astra_reader():
    asr_path = "notebooks/ASTRA_tutorial.pkl"
    astra = asr.ASTRAReader(asr_path)
    astra.print_metadata()
    astra.plot_recording()


def test_esc50_reader():
    esc_path = "notebooks/ESC50_tutorial_800Hz.pkl"
    esc800 = ecr.ESC50Reader(esc_path)
    esc800.print_metadata()
    esc800.plot_clip(0)


def test_orex_reader():
    orex_path = "notebooks/OREX_tutorial.pkl"
    orex_ds = oxr.OREXReader(orex_path)
    orex_ds.print_metadata()
    orex_ds.plot_all()


def test_shared_reader():
    shared_path = "notebooks/SHAReD_tutorial.pkl"
    shared_ds = ser.SHAReDReader(shared_path)
    shared_ds.print_metadata()
    shared_ds.plot_data()


if __name__ == "__main__":
    print("\nASTRA dataset reader test:")
    test_astra_reader()
    print("\nESC dataset reader test:")
    test_esc50_reader()
    print("\nOREX dataset reader test:")
    test_orex_reader()
    print("\nSHAReD dataset reader test:")
    test_shared_reader()
    plt.show()
