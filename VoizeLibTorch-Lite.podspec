Pod::Spec.new do |s|
    s.name             = 'VoizeLibTorch-Lite'
    s.version          = '2.1.0'
    s.authors          = 'voize GmbH'
    s.license          = { :type => 'BSD' }
    s.homepage         = 'https://github.com/voize-gmbh/pytorch-lite-builds'
    s.source           = { :http => "https://github.com/voize-gmbh/pytorch-lite-builds/releases/download/v2.1.0/libtorch_lite_ios_xc_2.1.0.zip" }
    s.summary          = 'The PyTorch C++ library for iOS'
    s.description      = <<-DESC
        The PyTorch C++ library for iOS.
    DESC

    s.ios.deployment_target = '12.0'

    s.vendored_frameworks = 'LibTorchLite.xcframework'

    s.pod_target_xcconfig = {
        'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
        'CLANG_CXX_LIBRARY' => 'libc++'
    }

    s.user_target_xcconfig = {
        'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
        'CLANG_CXX_LIBRARY' => 'libc++'
    }

    s.library = ['c++', 'stdc++']
    
    s.source_files = 'src/*.{h,cpp,c,cc}'
    s.public_header_files = ['src/LibTorch-Lite.h']

    s.frameworks = 'Accelerate'
end
