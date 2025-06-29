# Code Citations

## License: unknown
https://github.com/WarrenWeckesser/wavio/tree/296aba83c7775e0f500982ec6fb05d7823d46085/.github/workflows/python-publish.yml

```
runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python
```


## License: Apache_2_0
https://github.com/AgileRL/AgileRL/tree/6deb8c05dd346f938eadf0954b25b821a77e84c2/.github/workflows/python-app.yml

```
uses: actions/checkout@v4
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install
```


## License: MIT
https://github.com/lucienshawls/LibFetch_CPU/tree/8b003782219ac17a2c5d8a26b29bb7ec0c09ae94/.github/workflows/libfetch.yaml

```
: actions/checkout@v4
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -
```

