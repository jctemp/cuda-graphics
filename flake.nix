{
  description = "Cuda project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  };

  outputs = { self, nixpkgs, pre-commit-hooks }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      buildInputs = with pkgs; [
        cudatoolkit
        gnumake
        linuxPackages.nvidia_x11

        freeglut
        glew
        libGL
        libGLU
        libglvnd

        clang-tools
        cmake
        gcc11
      ];
      checks = {
        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixpkgs-fmt.enable = true;
            clang-format = {
              enable = true;
              types_or = [ "c" "c++" "cuda" ];
            };
            clang-tidy = {
              enable = true;
              types_or = [ "c" "c++" "cuda" ];
            };
          };
        };
      };
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;

      packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "cuda-graphics";
        src = ./.;
        inherit buildInputs;

        configurePhase = ''
          mkdir -p target 
          cmake -S . -B target
        '';

        preBuild = ''
          export CUDA_PATH="${pkgs.cudatoolkit}"
        '';

        buildPhase = ''
          cmake --build target 
        '';

        installPhase = ''
          mkdir -p $out/bin
          mv target/cuda-graphics $out/bin
        '';
      };

      devShells.${system}.default = pkgs.mkShell {
        inherit buildInputs;
        packages = with pkgs; [
          (python311.withPackages (pp: with pp; [
            numpy
            pandas
            matplotlib
            jupyter-core
            jupyter
            ipykernel
          ]))
        ];
        shellHook = checks.pre-commit-check.shellHook +
          ''
            export CUDA_PATH="${pkgs.cudatoolkit}"
          '';
      };
    };
}
