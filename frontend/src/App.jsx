import React, { useState } from "react";

function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [result,setResult] = useState(null);
  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("image1", image1);
    formData.append("image2", image2);
    console.log(image1);
    

    fetch("http://127.0.0.1:8000/home/", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (response.ok) {
          console.log("Images uploaded successfully");
          return response.json()
        } else {
          console.log("Upload failed");
        }
      })
      .then((data=>{
        console.log(data);
        setResult(data)
      }))
      .catch((error) => {
        console.error(error);
      });
  };
  if (result){
    return(<h1>Result : {result.result}% Similar </h1>)  
  }
  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor="image1">Image 1</label>
        <input
          type="file"
          id="image1"
          name="image1"
          accept="image/*"
          onChange={(event) => setImage1(event.target.files[0])}
        />
      </div>
      <div>
        <label htmlFor="image2">Image 2</label>
        <input
          type="file"
          id="image2"
          name="image2"
          accept="image/*"
          onChange={(event) => setImage2(event.target.files[0])}
        />
      </div>
      <button type="submit">Submit</button>
    </form>
  );
}

export default App;