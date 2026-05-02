from nih_multilabel_training import build_parser, run_training


def main() -> None:
    parser = build_parser(default_backbone_type="convnextv2")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
