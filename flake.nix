{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  };

  outputs = { self, nixpkgs }: 
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
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
  in
  {
    packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "cuda-graphics";
        src = ./.;
        inherit buildInputs;

        configurePhase = ''
          mkdir -p target 
          cmake -S . -B target
        '';

        preBuild = ''
          export CUDA_PATH="${pkgs.cudaPackages_12_0.cudatoolkit}"
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
        shellHook = ''
          export CUDA_PATH="${pkgs.cudaPackages_12_0.cudatoolkit}"
        '';
    };
  };
}
