{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  };

  outputs = { self, nixpkgs }: 
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
  in
  {
    packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "cuda-graphics";
        src = ./.;
        buildInputs = with pkgs; [
          cudatoolkit 
          gnumake 
          linuxPackages.nvidia_x11
          
          freeglut
          glew
          libGL
          libGLU
          libglvnd
        ];

        preBuild = ''
          export CUDA_PATH="${pkgs.cudatoolkit}"
        '';

        buildPhase = ''
          make
        '';

        installPhase = ''
          mkdir -p $out/bin
          cp cuda-graphics $out/bin/
        '';
    };    

    devShells.${system}.default = pkgs.mkShell {
      buildInputs = with pkgs; [
          cudaPackages_12_0.cudatoolkit
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
      shellHook = ''
        export IMAGE=nvidia/cuda:12.2.0-devel-ubuntu20.04
        export CUDA_PATH=${pkgs.cudaPackages_12_0.cudatoolkit}
      '';
    };
  };
}
